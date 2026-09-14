"""Cache successful builds by their inputs and verify artifact contents."""

import hashlib
import inspect
import json
import shutil
import tempfile
import time
from pathlib import Path


def digest(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def inventory(path: Path):
    return {
        str(entry.relative_to(path)): digest(entry)
        for entry in sorted(path.rglob("*"))
        if entry.is_file()
    }


def cache_key(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def artifact_identity(engine, corpus):
    adapter_source = inspect.getsourcefile(type(engine))
    return {
        "corpus_sha256": digest(corpus),
        "engine": getattr(engine, "build_identity", engine.identity)(),
        "adapter_source": digest(Path(adapter_source)) if adapter_source else None,
        "build_support": {
            name: digest(Path(__file__).with_name(name)) for name in ("dataset.py", "neural.py")
        },
        "cache_format": 2,
    }


def ensure_artifact(engine, corpus: Path, cache: Path):
    identity = artifact_identity(engine, corpus)
    key = cache_key(identity)
    cache.mkdir(parents=True, exist_ok=True)
    target, lock = cache / key, cache / f"{key}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Cache entry is locked: {lock}. Check its owning process.") from error
    try:
        record = target / "build.json"
        if record.exists():
            saved = json.loads(record.read_text())
            if saved["identity"] == identity and saved["files"] == inventory(target / "artifact"):
                return target / "artifact", {**saved, "cache_hit": True, "key": key}
            raise ValueError(
                f"Cached artifact is damaged: {target}. Remove this entry and rebuild."
            )
        if target.exists():
            raise ValueError(f"Cache entry is incomplete: {target}. Remove this entry and rebuild.")
        with tempfile.TemporaryDirectory(prefix=f"{key}-", dir=cache) as temporary:
            stage = Path(temporary)
            artifact = stage / "artifact"
            artifact.mkdir()
            started = time.perf_counter()
            engine.build(corpus, artifact)
            elapsed = time.perf_counter() - started
            if (
                digest(corpus) != identity["corpus_sha256"]
                or getattr(engine, "build_identity", engine.identity)() != identity["engine"]
            ):
                raise ValueError("Build inputs changed while the index was built.")
            files = inventory(artifact)
            if not files:
                raise ValueError("The engine did not create an artifact.")
            saved = {"identity": identity, "files": files, "build_seconds": elapsed}
            (stage / "build.json").write_text(json.dumps(saved, indent=2) + "\n")
            stage.rename(target)
        return target / "artifact", {**saved, "cache_hit": False, "key": key}
    finally:
        shutil.rmtree(lock)
