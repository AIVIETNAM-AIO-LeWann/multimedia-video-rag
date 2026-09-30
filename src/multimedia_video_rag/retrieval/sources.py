"""Download the parquet/safetensors/marker files an index build needs from the Hub."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from multimedia_video_rag.ingestion.audit import CONTRACTS

# Files each module contributes to an index build. Keyframe images are excluded:
# the index only needs identity/timestamps, and images are fetched on demand for display.
MODULE_FILES: dict[str, tuple[str, ...]] = {
    "keyframe": ("frames.parquet", "_SUCCESS.json"),
    "siglip": ("embeddings.safetensors", "frames.parquet", "_VISUAL_SUCCESS.json"),
    "beit3": ("embeddings.safetensors", "frames.parquet", "_VISUAL_SUCCESS.json"),
    "od": ("detections.parquet", "frames.parquet", "_OD_SUCCESS.json"),
    "asr": ("asr.parquet", "asr_chunks.parquet", "_ASR_SUCCESS.json"),
    "caption": ("captions.parquet", "_CAPTION_SUCCESS.json"),
}
REPOSITORIES: dict[str, str] = {
    contract.name: contract.repo_id for contract in CONTRACTS if contract.name in MODULE_FILES
}


def allow_patterns(module: str, video_ids: Iterable[str] | None = None) -> list[str]:
    videos = sorted(video_ids) if video_ids else ["*"]
    return [f"data/*/{video}/{name}" for video in videos for name in MODULE_FILES[module]]


def with_retries(
    operation: Callable[[], object],
    description: str,
    *,
    attempts: int = 8,
    base_delay: float = 30.0,
    max_delay: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Retry transient Hub failures (rate limits, 499/5xx, timeouts) with backoff.

    ``snapshot_download`` skips files already present in ``local_dir``, so each retry
    only re-requests what is still missing.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            print(
                f"{description}: {type(error).__name__}; retry {attempt}/{attempts - 1} "
                f"in {delay:.0f}s",
                flush=True,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def download_sources(
    cache_dir: Path,
    *,
    token: str | None,
    revision: str = "main",
    video_ids: Iterable[str] | None = None,
    max_workers: int = 4,
    completed: Mapping[str, Mapping[str, str]] | None = None,
    on_module_done: Callable[[dict[str, dict[str, str]]], None] | None = None,
) -> dict[str, dict[str, str]]:
    """Mirror every module into ``cache_dir/<module>`` pinned to one commit per repository.

    Modules listed in ``completed`` at the same commit are skipped without any Hub
    request per file; ``on_module_done`` receives the progress after each module so an
    interrupted run can resume from the next module.
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    selected = sorted(video_ids) if video_ids else None
    sources: dict[str, dict[str, str]] = {}
    for module, repo_id in REPOSITORIES.items():
        sha = with_retries(
            lambda repo_id=repo_id: api.dataset_info(repo_id, revision=revision).sha,
            f"resolve {repo_id}",
        )
        previous = (completed or {}).get(module, {})
        if previous.get("revision") == sha and previous.get("scope") == _scope(selected):
            print(f"Skipping {module}: already downloaded at {sha[:12]}", flush=True)
            sources[module] = dict(previous)
            continue
        print(f"Downloading {module} from {repo_id}@{sha[:12]} ...", flush=True)
        with_retries(
            lambda repo_id=repo_id, module=module, sha=sha: snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=sha,
                allow_patterns=allow_patterns(module, selected),
                local_dir=cache_dir / module,
                token=token,
                max_workers=max_workers,
            ),
            f"download {module}",
        )
        sources[module] = {"repo_id": repo_id, "revision": sha, "scope": _scope(selected)}
        if on_module_done is not None:
            on_module_done({**dict(completed or {}), **sources})
    return sources


def _scope(video_ids: list[str] | None) -> str:
    return "all" if video_ids is None else ",".join(video_ids)
