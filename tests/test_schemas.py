from pathlib import Path

import pandas as pd
import pytest

from multimedia_video_rag.ingestion.schemas import (
    KEYFRAME_COLUMNS,
    SchemaError,
    make_frame_uid,
    read_validated_parquet,
    require_one_video,
)


def test_frame_uid_is_stable_and_validated() -> None:
    assert make_frame_uid("L21_V001", 42) == "L21_V001:42"
    with pytest.raises(ValueError):
        make_frame_uid("video-1", 42)


def test_parquet_contract_reports_missing_columns(tmp_path: Path) -> None:
    artifact = tmp_path / "frames.parquet"
    pd.DataFrame({"video_id": ["L21_V001"]}).to_parquet(artifact)
    with pytest.raises(SchemaError, match="missing required columns"):
        read_validated_parquet(artifact, KEYFRAME_COLUMNS)


def test_empty_artifact_can_be_explicitly_valid(tmp_path: Path) -> None:
    artifact = tmp_path / "empty.parquet"
    assert require_one_video(pd.DataFrame(columns=["video_id"]), artifact, allow_empty=True) is None
