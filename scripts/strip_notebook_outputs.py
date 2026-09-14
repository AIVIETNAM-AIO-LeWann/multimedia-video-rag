"""Remove execution state from notebooks before version control."""

import json
import sys
from pathlib import Path
from typing import Any


def strip(path: Path) -> bool:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    metadata = document.get("metadata", {})
    if isinstance(metadata, dict) and metadata.pop("widgets", None) is not None:
        changed = True
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        cell_metadata = cell.get("metadata", {})
        colab = cell_metadata.get("colab", {}) if isinstance(cell_metadata, dict) else {}
        if isinstance(colab, dict) and colab.pop("referenced_widgets", None) is not None:
            changed = True
    if changed:
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def main(arguments: list[str]) -> int:
    paths = [Path(value) for value in arguments]
    if not paths:
        paths = sorted(Path("notebooks").rglob("*.ipynb"))
    for path in paths:
        status = "stripped" if strip(path) else "clean"
        print(f"{status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
