"""Use existing command-line engines and HTTP services without engine-specific runners."""

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib import request

from .cache import digest, inventory
from .dataset import identifier

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def fill(value, variables):
    if isinstance(value, str):
        for name, replacement in variables.items():
            if value == "{" + name + "}":
                return replacement
        return value
    if isinstance(value, list):
        return [fill(part, variables) for part in value]
    if isinstance(value, dict):
        return {key: fill(part, variables) for key, part in value.items()}
    return value


def field(value, path):
    for part in path.split(".") if path else []:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def hits(response, path, id_field):
    rows = field(response, path)
    if not isinstance(rows, list):
        raise ValueError("The selected response field must be a list.")
    return [identifier(field(row, id_field) if id_field else row) for row in rows]


class Command:
    boundary = (
        "one subprocess invocation per query, including process startup and result serialization"
    )

    def __init__(self, config):
        if set(config) - {
            "build",
            "search",
            "files",
            "identity",
            "timeout",
            "hits_path",
            "id_field",
        }:
            raise ValueError("Unknown command adapter configuration field.")
        if not isinstance(config.get("identity"), dict) or not config["identity"]:
            raise ValueError("Provide an identity object with engine and model versions.")
        self.config = config
        self.timeout = float(config.get("timeout", 3600))
        if not 0 < self.timeout <= 86400:
            raise ValueError("Command timeout must be positive and at most 86400 seconds.")
        for name in ("build", "search"):
            command = config.get(name)
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(v, str) for v in command)
            ):
                raise ValueError(f"{name} must be a nonempty argument list.")
            if shutil.which(command[0]) is None:
                raise ValueError(f"Executable not found: {command[0]}")
        if not isinstance(config.get("files", []), list):
            raise ValueError(
                "files must list scripts, configuration, and model files to fingerprint."
            )

    def identity(self):
        sources = {}
        for name in self.config.get("files", []):
            path = Path(name).resolve(strict=True)
            sources[str(path)] = inventory(path) if path.is_dir() else digest(path)
        return {
            "engine": "command",
            "configuration": self.config,
            "files": sources,
            "executables": {
                name: digest(Path(shutil.which(self.config[name][0])))
                for name in ("build", "search")
            },
        }

    def build(self, corpus, artifact):
        command = fill(self.config["build"], {"corpus": str(corpus), "artifact": str(artifact)})
        with tempfile.TemporaryFile(dir=artifact.parent) as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout, check=True
            )

    @contextmanager
    def open(self, artifact):
        def search(query, depth):
            command = fill(self.config["search"], {"artifact": str(artifact)})
            with tempfile.TemporaryFile(dir=artifact.parent) as output:
                with tempfile.TemporaryFile(dir=artifact.parent) as errors:
                    subprocess.run(
                        command,
                        input=json.dumps({"query": query, "depth": depth}).encode(),
                        stdout=output,
                        stderr=errors,
                        timeout=self.timeout,
                        check=True,
                    )
                output.seek(0)
                body = output.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Command response exceeds 16 MiB.")
            return hits(
                json.loads(body),
                self.config.get("hits_path", "hits"),
                self.config.get("id_field", ""),
            )

        yield search


class HTTP:
    boundary = (
        "HTTP round trip to an existing index, remote result-cache policy is caller controlled"
    )
    external_index = True

    def __init__(self, config):
        allowed = {"url", "request", "hits_path", "id_field", "headers_env", "identity", "timeout"}
        if (
            set(config) - allowed
            or not {"url", "request", "hits_path", "identity"} <= config.keys()
        ):
            raise ValueError("HTTP requires url, request, hits_path, and a versioned identity.")
        if not config["url"].startswith(("http://", "https://")):
            raise ValueError("Use an HTTP or HTTPS endpoint.")
        if not isinstance(config["identity"], dict) or not config["identity"]:
            raise ValueError("Identify the remote engine, model, corpus, and index revision.")
        if not {"engine", "revision", "corpus_sha256"} <= config["identity"].keys():
            raise ValueError("HTTP identity requires engine, revision, and corpus_sha256.")
        self.config = config
        self.timeout = float(config.get("timeout", 60))
        if not 0 < self.timeout <= 3600:
            raise ValueError("HTTP timeout must be positive and at most 3600 seconds.")

    def identity(self):
        # Credentials stay in the environment and do not enter reports or cache keys.
        return {
            "engine": "http",
            "configuration": self.config,
            "verification": "remote index identity supplied by caller",
        }

    def build(self, corpus, artifact):
        if digest(corpus) != self.config["identity"]["corpus_sha256"]:
            raise ValueError("The declared remote corpus hash does not match the benchmark corpus.")
        (artifact / "external-index.json").write_text(json.dumps(self.identity()) + "\n")

    @contextmanager
    def open(self, artifact):
        headers = {"content-type": "application/json"}
        headers.update(
            {key: os.environ[value] for key, value in self.config.get("headers_env", {}).items()}
        )

        def search(query, depth):
            body = fill(self.config["request"], {"query": query, "depth": depth})
            req = request.Request(
                self.config["url"], data=json.dumps(body).encode(), headers=headers
            )
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("HTTP response exceeds 16 MiB.")
            return hits(json.loads(raw), self.config["hits_path"], self.config.get("id_field", ""))

        yield search
