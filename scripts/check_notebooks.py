"""Validate canonical ingestion notebooks without executing GPU jobs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "ingestion"
EXPECTED = {
    "extract-kf-transnetv2.ipynb": ("aqpahm/aic2026-keyframes-transnetv2", "ByteDance/shot2story"),
    "ingest-asr-chunkformer-rnnt-large.ipynb": (
        "aqpahm/aic2026-asr-chunkformer-rnnt-large",
        "khanhld/chunkformer-rnnt-large-vie",
    ),
    "ingest-od-wedetect-large.ipynb": ("aqpahm/aic2026-od-wedetect-large", "WeDetect"),
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
    if found != set(EXPECTED):
        errors.append(f"notebook set mismatch: expected {sorted(EXPECTED)}, found {sorted(found)}")

    for name, required_strings in EXPECTED.items():
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
        for capability in ("GPU_IDS", "torch.cuda.device_count()", "ThreadPoolExecutor"):
            if capability not in source:
                errors.append(f"{name}: missing adaptive GPU capability {capability!r}")
        if TOKEN_PATTERN.search(source):
            errors.append(f"{name}: appears to contain a literal Hugging Face token")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("execution_count") is not None:
                errors.append(f"{name}: cell {index} has execution_count")
            if cell.get("outputs"):
                errors.append(f"{name}: cell {index} contains output")

    if errors:
        raise SystemExit("Notebook checks failed:\n- " + "\n- ".join(errors))
    print(f"Validated {len(EXPECTED)} clean ingestion notebooks.")


if __name__ == "__main__":
    main()
