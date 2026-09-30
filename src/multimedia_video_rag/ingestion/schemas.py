import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{3}$")

KEYFRAME_COLUMNS = frozenset(
    {"video_id", "sample_n", "shot_id", "frame_idx", "timestamp_sec", "image_path"}
)
VISUAL_IDENTITY_COLUMNS = frozenset(
    {
        "video_id",
        "frame_uid",
        "sample_n",
        "frame_idx",
        "shot_id",
        "timestamp_sec",
        "image_path",
        "embedding_row",
    }
)

CAPTION_COLUMNS = frozenset(
    {
        "video_id",
        "frame_uid",
        "sample_n",
        "frame_idx",
        "shot_id",
        "timestamp_sec",
        "image_path",
        "caption_en",
        "caption_en_normalized",
        "language",
        "model_id",
        "model_revision",
    }
)


class SchemaError(ValueError):
    """An artifact violates its declared ingestion contract."""


def make_frame_uid(video_id: str, frame_idx: int) -> str:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    return f"{video_id}:{frame_idx}"


def read_validated_parquet(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise SchemaError(f"{path}: missing required columns {sorted(missing)}")
    return frame


def require_one_video(frame: pd.DataFrame, path: Path, *, allow_empty: bool = False) -> str | None:
    if frame.empty:
        if allow_empty:
            return None
        raise SchemaError(f"{path}: expected at least one row")
    values = frame["video_id"].astype(str).unique().tolist()
    if len(values) != 1:
        raise SchemaError(f"{path}: expected one video_id, found {values[:5]}")
    video_id = str(values[0])
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise SchemaError(f"{path}: invalid video_id {video_id!r}")
    return video_id
