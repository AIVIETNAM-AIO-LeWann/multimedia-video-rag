"""Audit remote ingestion artifacts against the keyframe dataset inventory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from multimedia_video_rag.ingestion.schemas import (
    ASR_COLUMNS,
    KEYFRAME_COLUMNS,
    OD_COLUMNS,
    VISUAL_IDENTITY_COLUMNS,
    make_frame_uid,
)

KEYFRAME_REPO = "aqpahm/aic2026-keyframes-transnetv2"
DATA_PATH_RE = re.compile(r"^data/(L\d{2})/(L\d{2}_V\d{3})/([^/]+)$")

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
OD_FRAME_COLUMNS = frozenset(
    {
        "video_id",
        "frame_uid",
        "sample_n",
        "frame_idx",
        "shot_id",
        "timestamp_sec",
        "image_path",
        "detection_count",
        "status",
    }
)
ASR_CHUNK_COLUMNS = frozenset(
    {
        "video_id",
        "chunk_id",
        "start_sec",
        "end_sec",
        "timestamp_sec",
        "raw_text",
        "normalized_text",
        "normalized_no_accent",
        "language",
        "model_id",
        "model_revision",
    }
)


@dataclass(frozen=True)
class ModuleContract:
    name: str
    repo_id: str
    marker_name: str
    required_files: tuple[str, ...]
    schema_version: int
    model_ids: tuple[str, ...] = ()


CONTRACTS = (
    ModuleContract(
        "keyframe",
        KEYFRAME_REPO,
        "_SUCCESS.json",
        ("keyframes.tar", "frames.parquet", "_SUCCESS.json"),
        1,
    ),
    ModuleContract(
        "siglip",
        "aqpahm/aic2026-visual-siglip2-so400m",
        "_VISUAL_SUCCESS.json",
        ("embeddings.safetensors", "frames.parquet", "_VISUAL_SUCCESS.json"),
        1,
        ("google/siglip2-so400m-patch16-384",),
    ),
    ModuleContract(
        "beit3",
        "aqpahm/aic2026-visual-beit3-large-coco-retrieval",
        "_VISUAL_SUCCESS.json",
        ("embeddings.safetensors", "frames.parquet", "_VISUAL_SUCCESS.json"),
        1,
        ("microsoft/beit3-large-patch16-384-coco-retrieval",),
    ),
    ModuleContract(
        "od",
        "aqpahm/aic2026-od-wedetect-large",
        "_OD_SUCCESS.json",
        ("detections.parquet", "frames.parquet", "_OD_SUCCESS.json"),
        2,
        ("WeChatCV/WeDetect-large",),
    ),
    ModuleContract(
        "asr",
        "aqpahm/aic2026-asr-chunkformer-rnnt-large",
        "_ASR_SUCCESS.json",
        ("asr.parquet", "asr_chunks.parquet", "_ASR_SUCCESS.json"),
        3,
        ("khanhld/chunkformer-rnnt-large-vie", "Systran/faster-whisper-large-v3"),
    ),
    ModuleContract(
        "caption",
        "aqpahm/aic2026-caption-blip2-opt-2.7b-coco",
        "_CAPTION_SUCCESS.json",
        ("captions.parquet", "_CAPTION_SUCCESS.json"),
        1,
        ("Salesforce/blip2-opt-2.7b-coco",),
    ),
)


class HubClient(Protocol):
    def list_files(self, repo_id: str) -> list[str]: ...

    def download(self, repo_id: str, filename: str) -> Path: ...


class HuggingFaceHubClient:
    """Small adapter that keeps Hugging Face imports out of pure audit tests."""

    def __init__(self, *, token: str | None = None, revision: str = "main") -> None:
        from huggingface_hub import HfApi, hf_hub_download

        self._api = HfApi(token=token)
        self._download = hf_hub_download
        self._token = token
        self._revision = revision

    def list_files(self, repo_id: str) -> list[str]:
        return self._api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=self._revision,
            token=self._token,
        )

    def download(self, repo_id: str, filename: str) -> Path:
        return Path(
            self._download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=self._revision,
                token=self._token,
            )
        )


def inventory_from_files(files: Iterable[str]) -> tuple[dict[str, set[str]], list[str]]:
    """Return per-video basenames and malformed data paths."""
    inventory: dict[str, set[str]] = {}
    malformed: list[str] = []
    for filename in files:
        if not filename.startswith("data/"):
            continue
        match = DATA_PATH_RE.fullmatch(filename)
        if match is None:
            malformed.append(filename)
            continue
        level, video_id, basename = match.groups()
        if video_id.split("_")[0] != level:
            malformed.append(filename)
            continue
        inventory.setdefault(video_id, set()).add(basename)
    return inventory, sorted(malformed)


def remote_path(video_id: str, basename: str) -> str:
    return f"data/{video_id.split('_')[0]}/{video_id}/{basename}"


def read_marker(client: HubClient, contract: ModuleContract, video_id: str) -> dict[str, Any]:
    path = client.download(contract.repo_id, remote_path(video_id, contract.marker_name))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("marker root must be an object")
    return payload


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> list[str]:
    missing = sorted(set(required) - set(frame.columns))
    return [f"{label}: missing columns {missing}"] if missing else []


def _validate_video_column(
    frame: pd.DataFrame, video_id: str, label: str, *, allow_empty: bool = False
) -> list[str]:
    if frame.empty:
        return [] if allow_empty else [f"{label}: empty artifact"]
    values = sorted(frame["video_id"].astype(str).unique().tolist())
    return [] if values == [video_id] else [f"{label}: video_id values {values[:5]}"]


def _frame_uids(frame: pd.DataFrame) -> list[str]:
    if "frame_uid" in frame:
        return frame["frame_uid"].astype(str).tolist()
    return [make_frame_uid(str(row.video_id), int(row.frame_idx)) for row in frame.itertuples()]


def _validate_frame_alignment(frame: pd.DataFrame, source_uids: list[str], label: str) -> list[str]:
    uids = _frame_uids(frame)
    errors: list[str] = []
    if len(uids) != len(set(uids)):
        errors.append(f"{label}: duplicate frame_uid")
    if uids != source_uids:
        errors.append(
            f"{label}: frame identity/order mismatch "
            f"({len(uids)} rows vs {len(source_uids)} source)"
        )
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _catalog_hash(frame: pd.DataFrame, *, visual: bool) -> str:
    fields = [
        "video_id",
        "frame_uid",
        "sample_n",
        "frame_idx",
        "shot_id",
        "timestamp_sec",
        "image_path",
    ]
    if visual:
        fields.append("embedding_row")
    digest = hashlib.sha256()
    integer_fields = {"sample_n", "frame_idx", "shot_id", "embedding_row"}
    for values in frame.loc[:, fields].itertuples(index=False, name=None):
        payload: dict[str, Any] = {}
        for field, value in zip(fields, values, strict=True):
            if field in integer_fields:
                value = int(value)
            elif field == "timestamp_sec":
                value = float(value)
            payload[field] = value
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_marker(
    marker: Mapping[str, Any], contract: ModuleContract, video_id: str
) -> list[str]:
    errors: list[str] = []
    if marker.get("video_id") != video_id:
        errors.append(f"marker video_id={marker.get('video_id')!r}")
    try:
        schema_version = int(marker.get("schema_version", -1))
    except (TypeError, ValueError):
        schema_version = -1
    if schema_version != contract.schema_version:
        errors.append(f"marker schema_version={schema_version}, expected {contract.schema_version}")
    if marker.get("status") not in (None, "success", "no_speech"):
        errors.append(f"marker status={marker.get('status')!r}")
    if contract.model_ids and marker.get("model_id") not in contract.model_ids:
        errors.append(f"marker model_id={marker.get('model_id')!r}")
    if (
        contract.name in {"siglip", "beit3", "od", "caption"}
        and marker.get("source_repo") != KEYFRAME_REPO
    ):
        errors.append(f"marker source_repo={marker.get('source_repo')!r}")
    return errors


def _read_parquet(client: HubClient, contract: ModuleContract, video_id: str, name: str):
    path = client.download(contract.repo_id, remote_path(video_id, name))
    return path, pd.read_parquet(path)


def validate_parquets(
    client: HubClient,
    contract: ModuleContract,
    video_id: str,
    marker: Mapping[str, Any],
    source_uids: list[str] | None,
    *,
    verify_large_hashes: bool,
) -> tuple[list[str], int | None, list[str] | None]:
    """Validate one module and return errors, primary row count, and frame IDs."""
    errors: list[str] = []
    count: int | None = None
    uids: list[str] | None = None

    if contract.name == "keyframe":
        _, frame = _read_parquet(client, contract, video_id, "frames.parquet")
        errors += _validate_columns(frame, KEYFRAME_COLUMNS, "frames.parquet")
        if not errors:
            errors += _validate_video_column(frame, video_id, "frames.parquet")
            uids = _frame_uids(frame)
            count = len(frame)
            if len(uids) != len(set(uids)):
                errors.append("frames.parquet: duplicate frame_uid")
            if marker.get("keyframe_count") != count:
                errors.append(
                    f"marker keyframe_count={marker.get('keyframe_count')}, parquet rows={count}"
                )
        return errors, count, uids

    if contract.name in {"siglip", "beit3"}:
        _, frame = _read_parquet(client, contract, video_id, "frames.parquet")
        errors += _validate_columns(frame, VISUAL_IDENTITY_COLUMNS, "frames.parquet")
        if not errors:
            errors += _validate_video_column(frame, video_id, "frames.parquet")
            errors += _validate_frame_alignment(frame, source_uids or [], "frames.parquet")
            count = len(frame)
            uids = _frame_uids(frame)
            if frame["embedding_row"].astype(int).tolist() != list(range(count)):
                errors.append("frames.parquet: embedding_row is not contiguous from zero")
            if marker.get("frames") != count:
                errors.append(f"marker frames={marker.get('frames')}, parquet rows={count}")
            expected_dimension = 1152 if contract.name == "siglip" else 1024
            if marker.get("embedding_dimension") != expected_dimension:
                errors.append(
                    f"marker embedding_dimension={marker.get('embedding_dimension')}, "
                    f"expected {expected_dimension}"
                )
            if marker.get("catalog_sha256") != _catalog_hash(frame, visual=True):
                errors.append("marker catalog_sha256 mismatch")
        if verify_large_hashes:
            embedding = client.download(
                contract.repo_id, remote_path(video_id, "embeddings.safetensors")
            )
            if _sha256(embedding) != marker.get("embeddings_sha256"):
                errors.append("embeddings.safetensors: SHA-256 mismatch")
        return errors, count, uids

    if contract.name == "caption":
        path, frame = _read_parquet(client, contract, video_id, "captions.parquet")
        errors += _validate_columns(frame, CAPTION_COLUMNS, "captions.parquet")
        if not errors:
            errors += _validate_video_column(frame, video_id, "captions.parquet")
            errors += _validate_frame_alignment(frame, source_uids or [], "captions.parquet")
            count = len(frame)
            uids = _frame_uids(frame)
            empty = frame["caption_en"].isna() | frame["caption_en"].astype(str).str.strip().eq("")
            if empty.any():
                errors.append(f"captions.parquet: {int(empty.sum())} empty captions")
            if marker.get("frames") != count or marker.get("captions") != count:
                errors.append(
                    f"marker frames/captions={marker.get('frames')}/{marker.get('captions')}, "
                    f"parquet rows={count}"
                )
            if marker.get("catalog_sha256") != _catalog_hash(frame, visual=False):
                errors.append("marker catalog_sha256 mismatch")
            if marker.get("parquet_sha256") != _sha256(path):
                errors.append("captions.parquet: SHA-256 mismatch")
        return errors, count, uids

    if contract.name == "od":
        _, frames = _read_parquet(client, contract, video_id, "frames.parquet")
        _, detections = _read_parquet(client, contract, video_id, "detections.parquet")
        errors += _validate_columns(frames, OD_FRAME_COLUMNS, "frames.parquet")
        errors += _validate_columns(detections, OD_COLUMNS, "detections.parquet")
        if not errors:
            errors += _validate_video_column(frames, video_id, "frames.parquet")
            errors += _validate_video_column(
                detections, video_id, "detections.parquet", allow_empty=True
            )
            errors += _validate_frame_alignment(frames, source_uids or [], "frames.parquet")
            count = len(frames)
            uids = _frame_uids(frames)
            unknown = set(detections["frame_uid"].astype(str)) - set(source_uids or [])
            if unknown:
                errors.append(f"detections.parquet: {len(unknown)} unknown frame_uid values")
            if marker.get("frames") != count or marker.get("detections") != len(detections):
                errors.append(
                    f"marker frames/detections={marker.get('frames')}/{marker.get('detections')}, "
                    f"parquet rows={count}/{len(detections)}"
                )
        return errors, count, uids

    if contract.name == "asr":
        _, segments = _read_parquet(client, contract, video_id, "asr.parquet")
        _, chunks = _read_parquet(client, contract, video_id, "asr_chunks.parquet")
        errors += _validate_columns(segments, ASR_COLUMNS, "asr.parquet")
        errors += _validate_columns(chunks, ASR_CHUNK_COLUMNS, "asr_chunks.parquet")
        if not errors:
            errors += _validate_video_column(segments, video_id, "asr.parquet", allow_empty=True)
            errors += _validate_video_column(
                chunks, video_id, "asr_chunks.parquet", allow_empty=True
            )
            for label, frame in (("asr.parquet", segments), ("asr_chunks.parquet", chunks)):
                invalid = pd.to_numeric(frame["start_sec"], errors="coerce") > pd.to_numeric(
                    frame["end_sec"], errors="coerce"
                )
                if invalid.any():
                    errors.append(f"{label}: {int(invalid.sum())} start_sec > end_sec")
            count = len(segments)
            if marker.get("segments") != len(segments):
                errors.append(
                    f"marker segments={marker.get('segments')}, parquet rows={len(segments)}"
                )
            if marker.get("search_chunks") != len(chunks):
                errors.append(
                    f"marker search_chunks={marker.get('search_chunks')}, "
                    f"parquet rows={len(chunks)}"
                )
        return errors, count, None

    raise AssertionError(f"Unhandled module: {contract.name}")


def audit_repositories(
    client: HubClient,
    *,
    quick: bool = False,
    verify_large_hashes: bool = False,
    workers: int = 8,
    video_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit all configured repositories, using real keyframe members as truth."""
    repo_inventories: dict[str, dict[str, set[str]]] = {}
    malformed: dict[str, list[str]] = {}
    for contract in CONTRACTS:
        inventory, bad_paths = inventory_from_files(client.list_files(contract.repo_id))
        repo_inventories[contract.name] = inventory
        malformed[contract.name] = bad_paths

    all_source_videos = sorted(repo_inventories["keyframe"])
    source_set = set(all_source_videos)
    if video_ids is None:
        source_videos = all_source_videos
    else:
        requested = sorted(set(video_ids))
        unknown = sorted(set(requested) - source_set)
        if unknown:
            raise ValueError(f"video_ids are absent from keyframe inventory: {unknown}")
        source_videos = requested
    extra_videos = {
        contract.name: sorted(set(repo_inventories[contract.name]) - source_set)
        for contract in CONTRACTS
        if contract.name != "keyframe"
    }
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def audit_video(video_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {"video_id": video_id, "level": video_id.split("_")[0]}
        all_errors: list[str] = []
        source_uids: list[str] | None = None
        for contract in CONTRACTS:
            files = repo_inventories[contract.name].get(video_id, set())
            missing = sorted(set(contract.required_files) - files)
            errors = [f"missing files {missing}"] if missing else []
            count: int | None = None
            marker: dict[str, Any] | None = None
            if not errors and not quick:
                try:
                    marker = read_marker(client, contract, video_id)
                    errors += validate_marker(marker, contract, video_id)
                except Exception as exc:  # remote/data errors belong in the report
                    errors.append(f"marker: {type(exc).__name__}: {exc}")
            if marker is not None:
                try:
                    parquet_errors, count, discovered_uids = validate_parquets(
                        client,
                        contract,
                        video_id,
                        marker,
                        source_uids,
                        verify_large_hashes=verify_large_hashes,
                    )
                    errors += parquet_errors
                    if contract.name == "keyframe":
                        source_uids = discovered_uids
                except Exception as exc:  # keep auditing later videos/modules
                    errors.append(f"artifact: {type(exc).__name__}: {exc}")
            row[f"{contract.name}_status"] = "ok" if not errors else "error"
            row[f"{contract.name}_count"] = count
            row[f"{contract.name}_errors"] = json.dumps(errors, ensure_ascii=False)
            all_errors.extend(f"{contract.name}: {error}" for error in errors)
        row["status"] = "complete" if not all_errors else "error"
        row["errors"] = json.dumps(all_errors, ensure_ascii=False)
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(audit_video, source_videos))

    frame = pd.DataFrame(rows)
    complete = int((frame["status"] == "complete").sum()) if not frame.empty else 0
    summary = {
        "source_repo": KEYFRAME_REPO,
        "source_video_count": len(all_source_videos),
        "audited_video_count": len(source_videos),
        "complete_video_count": complete,
        "error_video_count": len(source_videos) - complete,
        "quick": quick,
        "verify_large_hashes": verify_large_hashes,
        "workers": workers,
        "repositories": {contract.name: contract.repo_id for contract in CONTRACTS},
        "repository_video_counts": {
            name: len(inventory) for name, inventory in repo_inventories.items()
        },
        "extra_videos": extra_videos,
        "malformed_data_paths": malformed,
    }
    return frame, summary


def write_report(frame: pd.DataFrame, summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "audit-videos.parquet", index=False)
    (output_dir / "audit-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
