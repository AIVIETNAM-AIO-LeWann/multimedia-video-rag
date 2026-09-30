"""Exercise HunyuanOCR output parsing and status rules without starting vLLM."""

import ast
import json
import re
import unicodedata
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/ingestion/ingest-ocr-hunyuanocr.ipynb"


def load_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    wanted = {
        "has_tail_repetition",
        "parse_spotting_output",
        "normalize_regions",
        "build_frame_result",
        "normalize_search",
        "remove_accents",
    }
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"json": json, "re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace, source


HELPERS, SOURCE = load_helpers()


def response(raw_output, finish_reason="stop"):
    return {
        "raw_output": raw_output,
        "finish_reason": finish_reason,
        "completion_tokens": 10,
        "latency_sec": 0.1,
    }


def frame_result(raw_output, finish_reason="stop"):
    return HELPERS["build_frame_result"](response(raw_output, finish_reason), 1280, 720, 1000, 8)


def test_notebook_pins_validated_vllm_stack_and_official_sampling():
    assert 'MODEL_ID = "tencent/HunyuanOCR"' in SOURCE
    assert 'OUTPUT_REPO = "aqpahm/aic2026-ocr-hunyuanocr"' in SOURCE
    assert 'REQUIRED_VLLM = "0.18.1"' in SOURCE
    assert "transformers==5" not in SOURCE
    assert "import torch" not in SOURCE
    assert 'ignore_patterns=["v1.0/*", "dflash/*", "assets/*"]' in SOURCE
    assert 'TASK_TYPE = "spotting_hunyuan"' in SOURCE
    official_prompt = "检测并识别图片中的文字，将文本坐标格式化输出。"  # noqa: RUF001
    assert f'"spotting_hunyuan": "{official_prompt}"' in SOURCE
    assert "TEMPERATURE = 0.0" in SOURCE
    assert "REPETITION_PENALTY = 1.08" in SOURCE
    assert '{"role": "system", "content": ""}' in SOURCE
    assert 'VLLM_DTYPE = "float16"' in SOURCE
    assert "RUN_SMOKE_TEST_BEFORE_INGEST = True" in SOURCE


def test_hunyuan_format_keeps_vietnamese_and_parentheses():
    raw = "Hà Nội (TTXVN)(100,50),(500,120)Thời sự 19h(20,900),(300,980)"
    result = frame_result(raw)
    assert result["ocr_status"] == "ok"
    assert result["spotting_format"] == "hunyuan"
    assert [region["text"] for region in result["regions"]] == ["Hà Nội (TTXVN)", "Thời sự 19h"]
    first = result["regions"][0]
    assert (first["bbox_x1"], first["bbox_y1"], first["bbox_x2"], first["bbox_y2"]) == (
        0.1,
        0.05,
        0.5,
        0.12,
    )
    assert result["ocr_text"] == "Hà Nội (TTXVN)\nThời sự 19h"
    assert [region["reading_order"] for region in result["regions"]] == [0, 1]


def test_decomposed_vietnamese_is_nfc_normalized():
    decomposed = unicodedata.normalize("NFD", "Việt Nam")
    result = frame_result(f"{decomposed}(0,0),(10,10)")
    assert result["regions"][0]["text"] == "Việt Nam"


def test_json_format_including_code_fence():
    raw = '```json\n[{"box": [0, 0, 500, 100], "text": "Bản tin \\"Thời sự\\""}]\n```'
    result = frame_result(raw)
    assert result["spotting_format"] == "json"
    assert result["regions"][0]["text"] == 'Bản tin "Thời sự"'


@pytest.mark.parametrize("raw", ["", "   ", "[]"])
def test_empty_output_is_no_text(raw):
    result = frame_result(raw)
    assert result["ocr_status"] == "no_text"
    assert result["regions"] == []
    assert result["ocr_text"] == ""


def test_unparseable_output_is_parse_error_not_no_text():
    result = frame_result("Hình ảnh không có văn bản rõ ràng")
    assert result["ocr_status"] == "parse_error"
    assert result["raw_output"] == "Hình ảnh không có văn bản rõ ràng"


def test_token_limit_is_truncated_even_with_parsed_regions():
    result = frame_result("Tin nóng(0,0),(100,100)Tin", finish_reason="length")
    assert result["ocr_status"] == "truncated"
    assert result["regions"][0]["text"] == "Tin nóng"


def test_tail_repetition_is_flagged():
    raw = "Giá xăng(0,0),(100,100)" + "abc" * 40
    assert HELPERS["has_tail_repetition"](raw)
    assert frame_result(raw)["ocr_status"] == "repetition"
    assert not HELPERS["has_tail_repetition"]("Giá xăng(0,0),(100,100)")


def test_invalid_boxes_are_clamped_or_dropped():
    raw = "A(0,0),(1200,100)B(10,10),(10,50)A(0,0),(1200,100)   (1,1),(2,2)"
    result = frame_result(raw)
    assert [region["text"] for region in result["regions"]] == ["A"]
    assert result["regions"][0]["bbox_x2"] == 1.0
    assert result["dropped_region_count"] == 2


def test_swapped_corners_are_reordered():
    result = frame_result("Chữ(500,400),(100,200)")
    region = result["regions"][0]
    assert (region["bbox_x1"], region["bbox_y1"], region["bbox_x2"], region["bbox_y2"]) == (
        0.1,
        0.2,
        0.5,
        0.4,
    )


def test_all_regions_invalid_is_parse_error():
    result = frame_result('[{"box": [10, 10, 10, 10], "text": "x"}]')
    assert result["ocr_status"] == "parse_error"
    assert result["dropped_region_count"] == 1


def test_search_normalization():
    normalized = HELPERS["normalize_search"]("  Đường   Trường Sơn ")
    assert normalized == "đường trường sơn"
    assert HELPERS["remove_accents"](normalized) == "duong truong son"
