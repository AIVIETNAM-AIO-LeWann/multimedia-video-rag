"""Build experimental FAISS + SQLite indexes from local copies of ingestion artifacts.

Every module is validated against the keyframe inventory: each video needs its success
marker, and frame-level modules must cover exactly the same ``frame_uid`` set. The index
directory is written to a ``.partial`` sibling and only renamed into place once complete.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multimedia_video_rag.ingestion.audit import (
    ASR_CHUNK_COLUMNS,
    CAPTION_COLUMNS,
    OD_FRAME_COLUMNS,
)
from multimedia_video_rag.ingestion.schemas import (
    ASR_COLUMNS,
    KEYFRAME_COLUMNS,
    OD_COLUMNS,
    VISUAL_IDENTITY_COLUMNS,
    SchemaError,
    make_frame_uid,
    read_validated_parquet,
)

INDEX_SCHEMA_VERSION = 1
VISUAL_MODULES = {"siglip": 1152, "beit3": 1024}
MARKERS = {
    "keyframe": "_SUCCESS.json",
    "siglip": "_VISUAL_SUCCESS.json",
    "beit3": "_VISUAL_SUCCESS.json",
    "od": "_OD_SUCCESS.json",
    "asr": "_ASR_SUCCESS.json",
    "caption": "_CAPTION_SUCCESS.json",
}
NORM_TOLERANCE = 1e-2
# Present in the WeDetect artifacts; kept for spatial queries ("xe bên trái").
OD_POSITION_COLUMNS = frozenset(
    {"center_x_norm", "center_y_norm", "position_horizontal", "position_vertical"}
)


@dataclass(frozen=True)
class VideoRef:
    level: str
    video_id: str

    def directory(self, module_root: Path) -> Path:
        return module_root / "data" / self.level / self.video_id


def discover_videos(keyframe_root: Path, video_ids: Iterable[str] | None = None) -> list[VideoRef]:
    """List keyframe videos that are present locally, in (level, video_id) order."""
    wanted = set(video_ids) if video_ids else None
    videos = [
        VideoRef(path.parent.parent.name, path.parent.name)
        for path in sorted((keyframe_root / "data").glob("L*/L*_V*/frames.parquet"))
    ]
    if wanted is not None:
        found = {video.video_id for video in videos}
        if missing := wanted - found:
            raise SchemaError(f"Requested videos missing from keyframe cache: {sorted(missing)}")
        videos = [video for video in videos if video.video_id in wanted]
    if not videos:
        raise SchemaError(f"No keyframe frames.parquet under {keyframe_root}")
    return videos


def require_markers(module: str, module_root: Path, videos: Iterable[VideoRef]) -> None:
    missing = [
        video.video_id
        for video in videos
        if not (video.directory(module_root) / MARKERS[module]).is_file()
    ]
    if missing:
        raise SchemaError(
            f"{module}: {len(missing)} video(s) lack a success marker: {missing[:10]}"
        )


def marker_models(module: str, module_root: Path, videos: Iterable[VideoRef]) -> dict[str, Any]:
    """Collect the model ids/revisions recorded in markers, so queries can use the same model."""
    model_ids, revisions = set(), set()
    for video in videos:
        path = video.directory(module_root) / MARKERS[module]
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("model_id"):
            model_ids.add(str(marker["model_id"]))
        if marker.get("model_revision"):
            revisions.add(str(marker["model_revision"]))
    return {"model_ids": sorted(model_ids), "model_revisions": sorted(revisions)}


def _check_single_video(frame: pd.DataFrame, video: VideoRef, label: str) -> None:
    values = set(frame["video_id"].astype(str))
    if values and values != {video.video_id}:
        raise SchemaError(
            f"{label} {video.video_id}: unexpected video_id values {sorted(values)[:5]}"
        )


def load_canonical_frames(keyframe_root: Path, videos: list[VideoRef]) -> pd.DataFrame:
    """One row per keyframe with a global ``frame_row`` id used by every index."""
    require_markers("keyframe", keyframe_root, videos)
    parts = []
    for video in videos:
        path = video.directory(keyframe_root) / "frames.parquet"
        frame = read_validated_parquet(path, KEYFRAME_COLUMNS)
        if frame.empty:
            raise SchemaError(f"{path}: no keyframes")
        _check_single_video(frame, video, "keyframe")
        frame = frame.sort_values("sample_n", kind="stable")
        if frame["frame_idx"].duplicated().any():
            raise SchemaError(f"{path}: duplicate frame_idx")
        parts.append(
            pd.DataFrame(
                {
                    "frame_uid": [
                        make_frame_uid(video.video_id, int(i)) for i in frame["frame_idx"]
                    ],
                    "video_id": video.video_id,
                    "level": video.level,
                    "sample_n": frame["sample_n"].astype("int64").to_numpy(),
                    "frame_idx": frame["frame_idx"].astype("int64").to_numpy(),
                    "shot_id": frame["shot_id"].astype("int64").to_numpy(),
                    "timestamp_sec": frame["timestamp_sec"].astype("float64").to_numpy(),
                    "image_path": frame["image_path"].astype(str).to_numpy(),
                }
            )
        )
    frames = pd.concat(parts, ignore_index=True)
    frames.insert(0, "frame_row", np.arange(len(frames), dtype=np.int64))
    return frames


def _frame_rows(video_frames: pd.DataFrame, uids: pd.Series, label: str) -> np.ndarray:
    """Map an artifact's frame_uid column onto canonical rows, requiring an exact match."""
    uid_list = uids.astype(str).tolist()
    if len(set(uid_list)) != len(uid_list):
        raise SchemaError(f"{label}: duplicate frame_uid")
    expected = set(video_frames["frame_uid"])
    if set(uid_list) != expected:
        extra = sorted(set(uid_list) - expected)[:5]
        missing = sorted(expected - set(uid_list))[:5]
        raise SchemaError(f"{label}: frame_uid mismatch; missing {missing}, extra {extra}")
    lookup = dict(zip(video_frames["frame_uid"], video_frames["frame_row"], strict=True))
    return np.array([lookup[uid] for uid in uid_list], dtype=np.int64)


def load_visual_matrix(
    module: str, module_root: Path, frames: pd.DataFrame, videos: list[VideoRef]
) -> np.ndarray:
    """Stack one module's FP16 embeddings in canonical ``frame_row`` order."""
    from safetensors.numpy import load_file

    require_markers(module, module_root, videos)
    dimension = VISUAL_MODULES[module]
    matrix = np.empty((len(frames), dimension), dtype=np.float16)
    filled = np.zeros(len(frames), dtype=bool)
    by_video = dict(tuple(frames.groupby("video_id", sort=False)))
    for video in videos:
        directory = video.directory(module_root)
        label = f"{module} {video.video_id}"
        identity = read_validated_parquet(directory / "frames.parquet", VISUAL_IDENTITY_COLUMNS)
        _check_single_video(identity, video, module)
        embeddings = load_file(str(directory / "embeddings.safetensors"))["embeddings"]
        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            raise SchemaError(f"{label}: expected (*, {dimension}), got {embeddings.shape}")
        source_rows = identity["embedding_row"].astype("int64").to_numpy()
        if len(source_rows) and (source_rows.min() < 0 or source_rows.max() >= len(embeddings)):
            raise SchemaError(f"{label}: embedding_row outside tensor bounds")
        rows = _frame_rows(by_video[video.video_id], identity["frame_uid"], label)
        matrix[rows] = embeddings[source_rows].astype(np.float16, copy=False)
        filled[rows] = True
    if not filled.all():
        raise SchemaError(f"{module}: {int((~filled).sum())} frame(s) without an embedding")
    # Check norms in blocks to avoid a full float32 copy of a ~335k-row matrix.
    worst = 0.0
    for start in range(0, len(matrix), 65536):
        block = matrix[start : start + 65536].astype(np.float32)
        if not np.isfinite(block).all():
            raise SchemaError(f"{module}: embeddings contain NaN or infinity")
        worst = max(worst, float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1.0))))
    if worst > NORM_TOLERANCE:
        raise SchemaError(f"{module}: embeddings are not L2-normalized (max |norm-1|={worst:.4f})")
    return matrix


def load_captions(root: Path, frames: pd.DataFrame, videos: list[VideoRef]) -> pd.DataFrame:
    require_markers("caption", root, videos)
    by_video = dict(tuple(frames.groupby("video_id", sort=False)))
    parts = []
    for video in videos:
        caption = read_validated_parquet(
            video.directory(root) / "captions.parquet", CAPTION_COLUMNS
        )
        _check_single_video(caption, video, "caption")
        rows = _frame_rows(
            by_video[video.video_id], caption["frame_uid"], f"caption {video.video_id}"
        )
        parts.append(
            pd.DataFrame(
                {
                    "frame_row": rows,
                    "caption_en": caption["caption_en"].astype(str).to_numpy(),
                    "model_id": caption["model_id"].astype(str).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values("frame_row", ignore_index=True)


def load_od(
    root: Path, frames: pd.DataFrame, videos: list[VideoRef]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_markers("od", root, videos)
    by_video = dict(tuple(frames.groupby("video_id", sort=False)))
    frame_parts, detection_parts = [], []
    for video in videos:
        directory = video.directory(root)
        video_frames = by_video[video.video_id]
        od_frames = read_validated_parquet(directory / "frames.parquet", OD_FRAME_COLUMNS)
        _check_single_video(od_frames, video, "od")
        rows = _frame_rows(video_frames, od_frames["frame_uid"], f"od {video.video_id}")
        detections = read_validated_parquet(
            directory / "detections.parquet", OD_COLUMNS | OD_POSITION_COLUMNS
        )
        _check_single_video(detections, video, "od detections")
        lookup = dict(zip(video_frames["frame_uid"], video_frames["frame_row"], strict=True))
        unknown = set(detections["frame_uid"].astype(str)) - set(lookup)
        if unknown:
            raise SchemaError(
                f"od {video.video_id}: detections on unknown frames {sorted(unknown)[:5]}"
            )
        detection_rows = detections["frame_uid"].astype(str).map(lookup).astype("int64")
        counts = detection_rows.value_counts()
        declared = pd.Series(od_frames["detection_count"].astype("int64").to_numpy(), index=rows)
        if not declared.eq(counts.reindex(declared.index, fill_value=0)).all():
            raise SchemaError(f"od {video.video_id}: detection_count does not match detections")
        frame_parts.append(
            pd.DataFrame(
                {
                    "frame_row": rows,
                    "detection_count": declared.to_numpy(),
                    "status": od_frames["status"].astype(str).to_numpy(),
                }
            )
        )
        detection_parts.append(
            pd.DataFrame(
                {
                    "frame_row": detection_rows.to_numpy(),
                    "class_id": detections["class_id"].astype("int64").to_numpy(),
                    "label_en": detections["label_en"].astype(str).to_numpy(),
                    "label_vi": detections["label_vi"].astype(str).to_numpy(),
                    "confidence": detections["confidence"].astype("float64").to_numpy(),
                    "x1": detections["x1_norm"].astype("float64").to_numpy(),
                    "y1": detections["y1_norm"].astype("float64").to_numpy(),
                    "x2": detections["x2_norm"].astype("float64").to_numpy(),
                    "y2": detections["y2_norm"].astype("float64").to_numpy(),
                    "box_area_ratio": detections["box_area_ratio"].astype("float64").to_numpy(),
                    "center_x": detections["center_x_norm"].astype("float64").to_numpy(),
                    "center_y": detections["center_y_norm"].astype("float64").to_numpy(),
                    "position_horizontal": (
                        detections["position_horizontal"].astype(str).to_numpy()
                    ),
                    "position_vertical": detections["position_vertical"].astype(str).to_numpy(),
                }
            )
        )
    od_frames = pd.concat(frame_parts, ignore_index=True)
    detections = pd.concat(detection_parts, ignore_index=True)
    return od_frames.sort_values("frame_row", ignore_index=True), detections


ASR_TEXT_COLUMNS = ["raw_text", "normalized_text", "normalized_no_accent", "model_id"]


def _load_asr_table(
    root: Path,
    videos: list[VideoRef],
    *,
    filename: str,
    required: frozenset[str],
    id_column: str,
    row_column: str,
) -> pd.DataFrame:
    parts = []
    for video in videos:
        table = read_validated_parquet(video.directory(root) / filename, required)
        _check_single_video(table, video, f"asr {filename}")
        if table.empty:
            continue  # The marker records a video without detected speech.
        start = table["start_sec"].astype("float64")
        end = table["end_sec"].astype("float64")
        if not (np.isfinite(start).all() and np.isfinite(end).all() and (end >= start).all()):
            raise SchemaError(f"asr {filename} {video.video_id}: invalid times")
        part = pd.DataFrame(
            {
                "video_id": video.video_id,
                id_column: table[id_column].astype(str).to_numpy(),
                "start_sec": start.to_numpy(),
                "end_sec": end.to_numpy(),
            }
        )
        for column in ASR_TEXT_COLUMNS:
            part[column] = table[column].astype(str).to_numpy()
        parts.append(part)
    columns = ["video_id", id_column, "start_sec", "end_sec", *ASR_TEXT_COLUMNS]
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    result.insert(0, row_column, np.arange(len(result), dtype=np.int64))
    return result


def load_asr(root: Path, videos: list[VideoRef]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (segments, search chunks). ASR stays interval-based; frames are joined by
    timestamp at query time. Chunks (~25 s windows with one-segment overlap) are the
    search unit designed at ingest; segments give finer localization."""
    require_markers("asr", root, videos)
    segments = _load_asr_table(
        root,
        videos,
        filename="asr.parquet",
        required=ASR_COLUMNS,
        id_column="segment_id",
        row_column="segment_row",
    )
    chunks = _load_asr_table(
        root,
        videos,
        filename="asr_chunks.parquet",
        required=ASR_CHUNK_COLUMNS,
        id_column="chunk_id",
        row_column="chunk_row",
    )
    with_speech = set(segments["video_id"])
    if with_speech != set(chunks["video_id"]):
        raise SchemaError("asr: videos with segments and videos with chunks differ")
    return segments, chunks


def write_faiss_index(matrix: np.ndarray, path: Path, *, chunk_rows: int = 65536) -> None:
    """Exact inner-product search over FP16-stored vectors (cosine, since rows are unit norm)."""
    import faiss

    index = faiss.IndexScalarQuantizer(
        matrix.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
    )
    for start in range(0, len(matrix), chunk_rows):
        index.add(np.ascontiguousarray(matrix[start : start + chunk_rows], dtype=np.float32))
    faiss.write_index(index, str(path))


SQLITE_SCHEMA = """
CREATE TABLE frames (
    frame_row INTEGER PRIMARY KEY,
    frame_uid TEXT NOT NULL UNIQUE,
    video_id TEXT NOT NULL,
    level TEXT NOT NULL,
    sample_n INTEGER NOT NULL,
    frame_idx INTEGER NOT NULL,
    shot_id INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    image_path TEXT NOT NULL
);
CREATE INDEX frames_video_time ON frames(video_id, timestamp_sec);
CREATE INDEX frames_video_shot ON frames(video_id, shot_id);

CREATE TABLE captions (
    frame_row INTEGER PRIMARY KEY REFERENCES frames(frame_row),
    caption_en TEXT NOT NULL,
    model_id TEXT NOT NULL
);
CREATE VIRTUAL TABLE captions_fts USING fts5(
    caption_en, content='captions', content_rowid='frame_row', tokenize='porter unicode61'
);

CREATE TABLE asr_segments (
    segment_row INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    normalized_no_accent TEXT NOT NULL,
    model_id TEXT NOT NULL
);
CREATE INDEX asr_video_time ON asr_segments(video_id, start_sec, end_sec);
CREATE VIRTUAL TABLE asr_fts USING fts5(
    normalized_text, normalized_no_accent,
    content='asr_segments', content_rowid='segment_row',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE asr_chunks (
    chunk_row INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    normalized_no_accent TEXT NOT NULL,
    model_id TEXT NOT NULL
);
CREATE INDEX asr_chunks_video_time ON asr_chunks(video_id, start_sec, end_sec);
CREATE VIRTUAL TABLE asr_chunks_fts USING fts5(
    normalized_text, normalized_no_accent,
    content='asr_chunks', content_rowid='chunk_row',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE od_frames (
    frame_row INTEGER PRIMARY KEY REFERENCES frames(frame_row),
    detection_count INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE od_detections (
    frame_row INTEGER NOT NULL REFERENCES frames(frame_row),
    class_id INTEGER NOT NULL,
    label_en TEXT NOT NULL,
    label_vi TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    box_area_ratio REAL NOT NULL,
    center_x REAL NOT NULL,
    center_y REAL NOT NULL,
    position_horizontal TEXT NOT NULL,
    position_vertical TEXT NOT NULL
);
CREATE INDEX od_detections_frame ON od_detections(frame_row);
"""

POST_LOAD_SQL = """
INSERT INTO captions_fts(captions_fts) VALUES('rebuild');
INSERT INTO asr_fts(asr_fts) VALUES('rebuild');
INSERT INTO asr_chunks_fts(asr_chunks_fts) VALUES('rebuild');
CREATE TABLE od_frame_labels AS
    SELECT frame_row, class_id, label_en, label_vi,
           COUNT(*) AS object_count, MAX(confidence) AS max_confidence
    FROM od_detections GROUP BY frame_row, class_id, label_en, label_vi;
CREATE INDEX od_frame_labels_label ON od_frame_labels(label_en, object_count);
CREATE INDEX od_frame_labels_frame ON od_frame_labels(frame_row);
"""


# sqlite3 cannot bind numpy integer scalars, which pandas can yield from int64 columns.
for _numpy_int in (np.int64, np.int32, np.int16, np.int8):
    sqlite3.register_adapter(_numpy_int, int)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.bool_, bool)


def _insert(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        frame.itertuples(index=False, name=None),
    )


def write_sqlite(
    path: Path,
    *,
    frames: pd.DataFrame,
    captions: pd.DataFrame,
    asr_segments: pd.DataFrame,
    asr_chunks: pd.DataFrame,
    od_frames: pd.DataFrame,
    detections: pd.DataFrame,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SQLITE_SCHEMA)
        with connection:
            _insert(connection, "frames", frames)
            _insert(connection, "captions", captions)
            _insert(connection, "asr_segments", asr_segments)
            _insert(connection, "asr_chunks", asr_chunks)
            _insert(connection, "od_frames", od_frames)
            _insert(connection, "od_detections", detections)
        connection.executescript(POST_LOAD_SQL)
        connection.commit()
    finally:
        connection.close()


def build_index(
    cache_dir: Path,
    output_dir: Path,
    *,
    sources: Mapping[str, Mapping[str, str]] | None = None,
    video_ids: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate local artifacts under ``cache_dir/<module>`` and publish an index directory."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")
    videos = discover_videos(cache_dir / "keyframe", video_ids)
    print(f"Videos: {len(videos)}", flush=True)
    frames = load_canonical_frames(cache_dir / "keyframe", videos)
    print(f"Keyframes: {len(frames)}", flush=True)
    matrices = {
        module: load_visual_matrix(module, cache_dir / module, frames, videos)
        for module in VISUAL_MODULES
    }
    captions = load_captions(cache_dir / "caption", frames, videos)
    od_frames, detections = load_od(cache_dir / "od", frames, videos)
    asr_segments, asr_chunks = load_asr(cache_dir / "asr", videos)
    visual_models = {
        module: marker_models(module, cache_dir / module, videos) for module in VISUAL_MODULES
    }
    print(
        f"Captions: {len(captions)} | OD detections: {len(detections)} | "
        f"ASR segments/chunks: {len(asr_segments)}/{len(asr_chunks)}",
        flush=True,
    )

    partial = output_dir.with_name(output_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for module, matrix in matrices.items():
        write_faiss_index(matrix, partial / f"{module}.faiss")
    write_sqlite(
        partial / "metadata.sqlite",
        frames=frames,
        captions=captions,
        asr_segments=asr_segments,
        asr_chunks=asr_chunks,
        od_frames=od_frames,
        detections=detections,
    )
    manifest = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "status": "experimental",
        "created_at": datetime.now(UTC).isoformat(),
        "sources": dict(sources or {}),
        "video_count": len(videos),
        "frame_count": len(frames),
        "caption_count": len(captions),
        "asr_segment_count": len(asr_segments),
        "asr_chunk_count": len(asr_chunks),
        "od_detection_count": len(detections),
        "faiss": {
            module: {
                "file": f"{module}.faiss",
                "dimension": int(matrix.shape[1]),
                "metric": "inner_product",
                "storage": "fp16",
                "id": "frame_row",
                **visual_models[module],
            }
            for module, matrix in matrices.items()
        },
        "sqlite": "metadata.sqlite",
    }
    (partial / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    partial.rename(output_dir)
    return manifest
