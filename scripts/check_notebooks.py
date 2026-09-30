"""Validate the notebooks in the repository without executing GPU jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "ingestion"
# Query-time helpers: only the hygiene checks apply (no outputs, counts or tokens).
RETRIEVAL_DIR = ROOT / "notebooks" / "retrieval"
EXPECTED = {
    "extract-kf-transnetv2.ipynb": ("aqpahm/aic2026-keyframes-transnetv2", "ByteDance/shot2story"),
    "ingest-visual-siglip2-so400m.ipynb": (
        "aqpahm/aic2026-visual-siglip2-so400m",
        "google/siglip2-so400m-patch16-384",
    ),
    "ingest-visual-beit3-large-coco-retrieval.ipynb": (
        "aqpahm/aic2026-visual-beit3-large-coco-retrieval",
        "beit3_large_patch16_384_coco_retrieval.pth",
    ),
    "ingest-caption-blip2-opt-2.7b-coco.ipynb": (
        "aqpahm/aic2026-caption-blip2-opt-2.7b-coco",
        "Salesforce/blip2-opt-2.7b-coco",
    ),
}
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def main() -> None:
    errors: list[str] = []
    found = {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")}
    expected_names = set(EXPECTED)
    if found != expected_names:
        errors.append(
            f"notebook set mismatch: expected {sorted(expected_names)}, found {sorted(found)}"
        )

    notebooks = EXPECTED
    for name, required_strings in notebooks.items():
        path = NOTEBOOK_DIR / name
        if not path.is_file():
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
            for cell in notebook.get("cells", [])
        )
        for value in required_strings:
            if value not in source:
                errors.append(f"{name}: missing expected value {value!r}")
        for runtime in ("colab", "kaggle"):
            if runtime not in source.lower():
                errors.append(f"{name}: missing {runtime} runtime support/documentation")
        if name in EXPECTED:
            for capability in ("GPU_IDS", "ThreadPoolExecutor"):
                if capability not in source:
                    errors.append(f"{name}: missing adaptive GPU capability {capability!r}")
            gpu_count_apis = (
                "torch.cuda.device_count()",
                "paddle.device.cuda.device_count()",
                "nvidia-smi",
            )
            if not any(value in source for value in gpu_count_apis):
                errors.append(f"{name}: missing adaptive GPU count API")
        if TOKEN_PATTERN.search(source):
            errors.append(f"{name}: appears to contain a literal Hugging Face token")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("execution_count") is not None:
                errors.append(f"{name}: cell {index} has execution_count")
            if cell.get("outputs"):
                errors.append(f"{name}: cell {index} contains output")

    retrieval = sorted(RETRIEVAL_DIR.glob("*.ipynb"))
    for path in retrieval:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        if TOKEN_PATTERN.search(source):
            errors.append(f"{path.name}: appears to contain a literal Hugging Face token")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("execution_count") is not None:
                errors.append(f"{path.name}: cell {index} has execution_count")
            if cell.get("outputs"):
                errors.append(f"{path.name}: cell {index} contains output")

    if errors:
        raise SystemExit("Notebook checks failed:\n- " + "\n- ".join(errors))
    print(f"Validated {len(EXPECTED)} ingestion and {len(retrieval)} retrieval notebooks.")


if __name__ == "__main__":
    main()
