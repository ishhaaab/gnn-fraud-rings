"""Execute code cells from the lightweight project notebook without Jupyter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def execute_notebook(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("nbformat") != 4 or not isinstance(document.get("cells"), list):
        raise ValueError(f"{path} is not a valid v4 notebook document")
    namespace = {"__name__": "__notebook__"}
    executed = 0
    for index, cell in enumerate(document["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if not isinstance(source, list) or not all(isinstance(line, str) for line in source):
            raise ValueError(f"code cell {index} has an invalid source")
        exec(compile("".join(source), f"{path}:cell-{index}", "exec"), namespace)
        executed += 1
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    count = execute_notebook(args.path)
    print(json.dumps({"notebook": str(args.path), "code_cells_executed": count}))


if __name__ == "__main__":
    main()
