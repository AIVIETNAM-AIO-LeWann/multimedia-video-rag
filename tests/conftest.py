"""Shared synthetic ingestion artifacts for retrieval tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimedia_video_rag.retrieval.build import MARKERS, VISUAL_MODULES

try:
    import safetensors.numpy as safetensors_numpy
except ImportError:  # Only the retrieval fixtures need it; those tests skip without it.
    safetensors_numpy = None

VIDEOS = {"L21_V001": ("L21", [0, 30, 90]), "L22_V004": ("L22", [5, 60])}


def unit_rows(count: int, dimension: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(count, dimension)).astype(np.float32)
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float16)


def write_marker(directory: Path, module: str, **fields: str) -> None:
    payload = {"status": "success", **fields}
    (directory / MARKERS[module]).write_text(json.dumps(payload), encoding="utf-8")


def identity(video_id: str, frame_indices: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": video_id,
            "frame_uid": [f"{video_id}:{index}" for index in frame_indices],
            "sample_n": range(1, len(frame_indices) + 1),
            "frame_idx": frame_indices,
            "shot_id": [0] * len(frame_indices),
            "timestamp_sec": [index / 25 for index in frame_indices],
            "image_path": [f"{video_id}/{n:04d}.jpg" for n in range(1, len(frame_indices) + 1)],
        }
    )


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    root = tmp_path / "hf"
    for seed, (video_id, (level, frame_indices)) in enumerate(VIDEOS.items()):
        ids = identity(video_id, frame_indices)

        directory = root / "keyframe" / "data" / level / video_id
        directory.mkdir(parents=True)
        ids.drop(columns="frame_uid").to_parquet(directory / "frames.parquet")
        write_marker(directory, "keyframe")

        for offset, (module, dimension) in enumerate(VISUAL_MODULES.items()):
            directory = root / module / "data" / level / video_id
            directory.mkdir(parents=True)
            vectors = unit_rows(len(ids), dimension, seed * 10 + offset)
            # Store rows reversed so the build must honor embedding_row.
            order = np.arange(len(ids))[::-1]
            safetensors_numpy.save_file(
                {"embeddings": vectors[order]}, str(directory / "embeddings.safetensors")
            )
            ids.assign(embedding_row=np.argsort(order)).to_parquet(directory / "frames.parquet")
            np.save(directory / "expected.npy", vectors)
            write_marker(directory, module, model_id=f"org/{module}", model_revision="rev1")

        directory = root / "caption" / "data" / level / video_id
        directory.mkdir(parents=True)
        ids.assign(
            caption_en=[f"a red bus on street {video_id} {i}" for i in range(len(ids))],
            caption_en_normalized="",
            language="en",
            model_id="Salesforce/blip2-opt-2.7b-coco",
            model_revision="x",
        ).to_parquet(directory / "captions.parquet")
        write_marker(directory, "caption")

        directory = root / "od" / "data" / level / video_id
        directory.mkdir(parents=True)
        first_uid = ids["frame_uid"].iloc[0]
        detections = pd.DataFrame(
            {
                "video_id": video_id,
                "frame_uid": [first_uid, first_uid],
                "frame_idx": [frame_indices[0]] * 2,
                "timestamp_sec": [0.0, 0.0],
                "class_id": [7, 7],
                "label_en": ["car", "car"],
                "label_vi": ["xe ô tô", "xe ô tô"],
                "label_zh": ["汽车", "汽车"],
                "confidence": [0.9, 0.6],
                "x1_norm": [0.1, 0.5],
                "y1_norm": [0.1, 0.5],
                "x2_norm": [0.3, 0.7],
                "y2_norm": [0.3, 0.7],
                "box_area_ratio": [0.04, 0.04],
                "center_x_norm": [0.2, 0.6],
                "center_y_norm": [0.2, 0.6],
                "position_horizontal": ["left", "center"],
                "position_vertical": ["top", "middle"],
            }
        )
        detections.to_parquet(directory / "detections.parquet")
        ids.assign(
            detection_count=[2] + [0] * (len(ids) - 1),
            status=["ok"] + ["empty_valid"] * (len(ids) - 1),
        ).to_parquet(directory / "frames.parquet")
        write_marker(directory, "od")

        directory = root / "asr" / "data" / level / video_id
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "video_id": [video_id],
                "segment_id": [0],
                "start_sec": [0.0],
                "end_sec": [4.5],
                "timestamp_sec": [0.0],
                "raw_text": ["Thời sự Việt Nam"],
                "normalized_text": ["thời sự việt nam"],
                "normalized_no_accent": ["thoi su viet nam"],
                "language": ["vi"],
                "model_id": ["khanhld/chunkformer-rnnt-large-vie"],
            }
        ).to_parquet(directory / "asr.parquet")
        pd.DataFrame(
            {
                "video_id": [video_id],
                "chunk_id": [f"{video_id}__asr_chunk_00000"],
                "start_sec": [0.0],
                "end_sec": [4.5],
                "timestamp_sec": [2.25],
                "raw_text": ["Thời sự Việt Nam"],
                "normalized_text": ["thời sự việt nam"],
                "normalized_no_accent": ["thoi su viet nam"],
                "source_segment_start": [f"{video_id}__asr_00000"],
                "source_segment_end": [f"{video_id}__asr_00000"],
                "language": ["vi"],
                "model_id": ["khanhld/chunkformer-rnnt-large-vie"],
                "model_revision": ["x"],
            }
        ).to_parquet(directory / "asr_chunks.parquet")
        write_marker(directory, "asr")
    return root
