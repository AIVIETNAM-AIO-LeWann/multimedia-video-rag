"""Resume and retry behaviour of the Hub mirror, without network access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from multimedia_video_rag.retrieval import sources
from multimedia_video_rag.retrieval.sources import REPOSITORIES, download_sources, with_retries


def test_with_retries_recovers_from_transient_errors():
    calls, delays = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("499")
        return "ok"

    assert with_retries(flaky, "test", base_delay=1, sleep=delays.append) == "ok"
    assert delays == [1, 2]


def test_with_retries_gives_up():
    with pytest.raises(RuntimeError, match="always"):
        with_retries(
            lambda: (_ for _ in ()).throw(RuntimeError("always")),
            "test",
            attempts=3,
            sleep=lambda _: None,
        )


@pytest.fixture
def fake_hub(monkeypatch):
    downloads: list[str] = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo_id, revision="main"):
            return SimpleNamespace(sha=f"sha-{repo_id}")

    def fake_snapshot_download(*, repo_id, **kwargs):
        downloads.append(repo_id)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    return downloads


def test_resume_skips_modules_completed_at_same_commit(fake_hub, tmp_path: Path):
    completed = {
        "keyframe": {
            "repo_id": REPOSITORIES["keyframe"],
            "revision": f"sha-{REPOSITORIES['keyframe']}",
            "scope": "all",
        },
        # Downloaded for a smoke subset only: must not count as a full download.
        "siglip": {
            "repo_id": REPOSITORIES["siglip"],
            "revision": f"sha-{REPOSITORIES['siglip']}",
            "scope": "L21_V001",
        },
    }
    progress = []
    result = download_sources(
        tmp_path, token=None, completed=completed, on_module_done=progress.append
    )

    assert REPOSITORIES["keyframe"] not in fake_hub
    assert REPOSITORIES["siglip"] in fake_hub
    assert set(result) == set(REPOSITORIES)
    assert all(entry["scope"] == "all" for entry in result.values())
    assert set(progress[-1]) == set(REPOSITORIES)


def test_changed_commit_downloads_again(fake_hub, tmp_path: Path):
    completed = {
        "keyframe": {"repo_id": REPOSITORIES["keyframe"], "revision": "old", "scope": "all"}
    }
    download_sources(tmp_path, token=None, completed=completed)
    assert REPOSITORIES["keyframe"] in fake_hub
