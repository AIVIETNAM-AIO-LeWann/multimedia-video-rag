"""Exercise the notebook's recovery code without GPU inference or Hub writes."""

import ast
import json
import re
import shutil
import unicodedata
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/ingestion/ingest-asr-chunkformer-rnnt-large.ipynb"
)


@pytest.fixture
def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text())
    trees = [
        ast.parse("".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and not "".join(cell["source"]).startswith("!")
    ]
    definitions = []
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"RAW_COLUMNS", "CHUNK_COLUMNS"}
                    for target in node.targets
                )
            ):
                definitions.append(node)
    namespace = {
        "Path": Path,
        "re": re,
        "pd": pd,
        "json": json,
        "shutil": shutil,
        "unicodedata": unicodedata,
        "datetime": datetime,
        "timezone": timezone,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace, trees[-1]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Videos_L21/L21_V001.mp4", True),
        ("L30_V123.MP4", True),
        ("L21_V001xmp4", False),
        ("L21_V001.mp4.bak", False),
        ("Ldd_Vddd.mp4", False),
        ("README.md", False),
    ],
)
def test_archive_member_filter(notebook_code, name, expected):
    namespace, _ = notebook_code
    assert namespace["is_video_member"](name) is expected


@pytest.mark.parametrize("with_speech", [True, False])
def test_recovery_of_two_consecutive_videos(notebook_code, tmp_path, with_speech):
    namespace, recovery_tree = notebook_code
    archive_path = tmp_path / "Videos_L21_a.zip"
    video_ids = {"L21_V001", "L21_V002"}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for video_id in sorted(video_ids):
            archive.writestr(f"videos/{video_id}.mp4", b"mock video")

    primary = Mock()
    primary.to.return_value = primary
    primary.device_name = "cuda:0"

    def move_primary(device):
        primary.device_name = device
        return primary

    primary.to.side_effect = move_primary

    def empty_primary(video_path, active_model):
        assert video_path.is_file()
        assert active_model is primary
        assert primary.device_name == "cuda:0"
        assert namespace["whisper_model"] is None
        raise RuntimeError("ChunkFormer returned no speech segments")

    segment = SimpleNamespace(text="Xin chào", start=1.0, end=2.0)
    fallback = Mock()
    fallback.transcribe.return_value = (
        [segment] if with_speech else [],
        SimpleNamespace(language="vi", language_probability=1.0),
    )

    def load_whisper(*args, **kwargs):
        assert primary.device_name == "cpu"
        return fallback

    loader = Mock(side_effect=load_whisper)
    namespace.update(
        OUTPUT_DIR=tmp_path / "output",
        EXTRACT_DIR=tmp_path / "extracted",
        RECOVERY_TARGETS={archive_path.name: video_ids},
        download_archive=lambda name: archive_path,
        zipfile=zipfile,
        probe_audio=lambda path: {"duration_sec": 10.0},
        transcribe_video=Mock(side_effect=empty_primary),
        models=[primary],
        DEVICES=["cuda:0"],
        GPU_IDS=[0],
        torch=SimpleNamespace(
            cuda=SimpleNamespace(device=lambda _: nullcontext(), empty_cache=Mock())
        ),
        gc=SimpleNamespace(collect=Mock()),
        WhisperModel=loader,
        whisper_model=None,
        FALLBACK_MODEL_ID="test/whisper",
        FALLBACK_DEVICE_INDEX=0,
        FALLBACK_COMPUTE_TYPE="float16",
        MODEL_ID="test/chunkformer",
        MODEL_REVISION="primary-revision",
        importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda _: "test-version")),
        recovery_files=[],
        recovery_report=[],
        recovered_ids=set(),
    )
    # Execute the actual recovery loop, so missing arguments/globals fail here.
    loop = next(
        node
        for node in recovery_tree.body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "RECOVERY_TARGETS"
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)

    assert namespace["recovered_ids"] == video_ids
    assert namespace["models"] == [primary]
    assert namespace["transcribe_video"].call_count == 2
    assert loader.call_count == 2
    assert fallback.transcribe.call_count == (2 if with_speech else 4)
    assert len(namespace["recovery_files"]) == 6
    for video_id in video_ids:
        directory = namespace["OUTPUT_DIR"] / "L21" / video_id
        raw = pd.read_parquet(directory / "asr.parquet")
        chunks = pd.read_parquet(directory / "asr_chunks.parquet")
        marker = json.loads((directory / "_ASR_SUCCESS.json").read_text())
        assert list(raw.columns) == namespace["RAW_COLUMNS"]
        assert list(chunks.columns) == namespace["CHUNK_COLUMNS"]
        assert len(raw) == len(chunks) == int(with_speech)
        assert marker["status"] == ("success" if with_speech else "no_detected_speech")
        assert marker["segments"] == int(with_speech)
        assert marker["schema_version"] == 3


def test_primary_retry_success_passes_model(notebook_code):
    namespace, _ = notebook_code
    model = Mock()
    model.to.return_value = model
    transcribe = Mock(return_value=([{"raw_text": "Xin chào"}], {"duration_sec": 2}))
    namespace.update(
        whisper_model=Mock(),
        models=[model],
        DEVICES=["cuda:0"],
        GPU_IDS=[0],
        torch=SimpleNamespace(
            cuda=SimpleNamespace(device=lambda _: nullcontext(), empty_cache=Mock())
        ),
        gc=SimpleNamespace(collect=Mock()),
        transcribe_video=transcribe,
    )
    video_path = Path("L21_V001.mp4")
    rows, info = namespace["retry_primary_transcription"](video_path)
    transcribe.assert_called_once_with(video_path, model)
    assert namespace["whisper_model"] is None
    assert rows and info["duration_sec"] == 2
