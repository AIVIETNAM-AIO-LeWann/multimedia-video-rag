"""Exercise PP-OCRv6 result parsing without loading PaddlePaddle models."""

import ast
import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/ingestion/ingest-ocr-ppocrv6-medium.ipynb"
)


def load_ocr_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    wanted_functions = {
        "result_payload",
        "polygon_to_region",
        "reading_order_key",
        "normalize_search",
        "remove_accents",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions
    ]
    namespace = {"json": json, "np": np, "unicodedata": unicodedata}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace, source


def test_notebook_uses_ppocrv6_without_generative_fallbacks():
    _, source = load_ocr_helpers()
    assert 'DETECTION_MODEL = "PP-OCRv6_medium_det"' in source
    assert 'RECOGNITION_MODEL = "PP-OCRv6_medium_rec"' in source
    assert 'OUTPUT_REPO = "aqpahm/aic2026-ocr-ppocrv6-medium"' in source
    assert 'DETECTION_MODEL_ID = "PaddlePaddle/PP-OCRv6_medium_det"' in source
    assert 'RECOGNITION_MODEL_ID = "PaddlePaddle/PP-OCRv6_medium_rec"' in source
    assert "snapshot_download(" in source
    assert "DETECTION_MODEL_REVISION" in source
    assert "RECOGNITION_MODEL_REVISION" in source
    assert "paddle.device.cuda.device_count()" in source
    assert "import torch" not in source
    assert "text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE" in source
    assert 'device=f"gpu:{gpu_id}"' in source
    assert "RUN_SMOKE_TEST_BEFORE_INGEST = True" in source
    assert "finalize_upload" in source
    assert "RUN_CONFIG_SHA256" in source
    assert "ocr_regions.parquet" in source
    for forbidden in (
        "AutoModel",
        "AutoProcessor",
        "generate(",
        "MAX_NEW_TOKENS",
        "TailRepetitionStop",
        "run_tile_retry",
        "SPOTTING_PROMPT",
    ):
        assert forbidden not in source


def test_polygon_is_normalized_and_preserves_vietnamese_text():
    namespace, _ = load_ocr_helpers()
    region = namespace["polygon_to_region"](
        "Tiếng Việt",
        0.93,
        [[100, 50], [500, 50], [500, 150], [100, 150]],
        1000,
        500,
    )
    assert region["text"] == "Tiếng Việt"
    assert region["confidence"] == 0.93
    assert region["bbox_x1"] == 0.1
    assert region["bbox_y1"] == 0.1
    assert region["bbox_x2"] == 0.5
    assert region["bbox_y2"] == 0.3
    assert json.loads(region["polygon_json"])[0] == [0.1, 0.1]


def test_result_payload_accepts_paddlex_result_json_property():
    namespace, _ = load_ocr_helpers()
    result = SimpleNamespace(json={"res": {"rec_texts": ["Xin chào"]}})
    assert namespace["result_payload"](result)["rec_texts"] == ["Xin chào"]


def test_reading_order_and_search_normalization():
    namespace, _ = load_ocr_helpers()
    top_right = {"bbox_x1": 0.6, "bbox_y1": 0.1, "bbox_x2": 0.9, "bbox_y2": 0.2}
    top_left = {"bbox_x1": 0.1, "bbox_y1": 0.1, "bbox_x2": 0.4, "bbox_y2": 0.2}
    bottom = {"bbox_x1": 0.1, "bbox_y1": 0.5, "bbox_x2": 0.4, "bbox_y2": 0.6}
    ordered = sorted([bottom, top_right, top_left], key=namespace["reading_order_key"])
    assert ordered == [top_left, top_right, bottom]
    normalized = namespace["normalize_search"]("  Việt   Nam  ")
    assert normalized == "việt nam"
    assert namespace["remove_accents"](normalized) == "viet nam"
