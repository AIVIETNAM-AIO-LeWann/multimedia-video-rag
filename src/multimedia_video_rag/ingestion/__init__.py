"""Artifact schemas and validation helpers for the ingested keyframe/embedding data."""

from multimedia_video_rag.ingestion.schemas import (
    CAPTION_COLUMNS,
    KEYFRAME_COLUMNS,
    VISUAL_IDENTITY_COLUMNS,
    SchemaError,
    make_frame_uid,
)

__all__ = [
    "CAPTION_COLUMNS",
    "KEYFRAME_COLUMNS",
    "VISUAL_IDENTITY_COLUMNS",
    "SchemaError",
    "make_frame_uid",
]
