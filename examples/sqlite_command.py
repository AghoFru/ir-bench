"""Example process adapter around a real SQLite FTS5 index."""

import json
import sys
from pathlib import Path

from ir_bench.adapters import SQLiteFTS5

engine = SQLiteFTS5({})
if sys.argv[1] == "build":
    engine.build(Path(sys.argv[2]), Path(sys.argv[3]))
elif sys.argv[1] == "search":
    request = json.load(sys.stdin)
    with engine.open(Path(sys.argv[2])) as search:
        print(json.dumps({"hits": search(request["query"], request["depth"])}))
else:
    raise SystemExit("Use build or search.")
