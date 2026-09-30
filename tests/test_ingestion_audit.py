import json
from pathlib import Path

from multimedia_video_rag.ingestion.audit import (
    CONTRACTS,
    KEYFRAME_REPO,
    ModuleContract,
    audit_repositories,
    inventory_from_files,
    validate_marker,
)


def test_inventory_uses_real_members_and_rejects_malformed_paths() -> None:
    inventory, malformed = inventory_from_files(
        [
            "README.md",
            "data/L21/L21_V001/frames.parquet",
            "data/L21/L21_V001/_SUCCESS.json",
            "data/L22/L21_V002/frames.parquet",
            "data/L21/L21_V003/nested/file.parquet",
        ]
    )

    assert inventory == {"L21_V001": {"frames.parquet", "_SUCCESS.json"}}
    assert malformed == [
        "data/L21/L21_V003/nested/file.parquet",
        "data/L22/L21_V002/frames.parquet",
    ]


def test_marker_validation_checks_identity_schema_model_and_source() -> None:
    contract = ModuleContract(
        "caption",
        "owner/caption",
        "_CAPTION_SUCCESS.json",
        ("captions.parquet", "_CAPTION_SUCCESS.json"),
        1,
        ("expected/model",),
    )
    errors = validate_marker(
        {
            "video_id": "L21_V999",
            "schema_version": 2,
            "status": "failed",
            "model_id": "wrong/model",
            "source_repo": "wrong/source",
        },
        contract,
        "L21_V001",
    )

    assert len(errors) == 5


class FakeHubClient:
    def __init__(self, root: Path, files: dict[str, list[str]]) -> None:
        self.root = root
        self.files = files

    def list_files(self, repo_id: str) -> list[str]:
        return self.files[repo_id]

    def download(self, repo_id: str, filename: str) -> Path:
        return self.root / repo_id.replace("/", "--") / filename


def _write(root: Path, repo_id: str, filename: str, content: str = "") -> Path:
    path = root / repo_id.replace("/", "--") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _marker(contract, video_id: str) -> dict:
    marker = {"video_id": video_id, "schema_version": contract.schema_version}
    if contract.model_ids:
        marker["model_id"] = contract.model_ids[0]
    if contract.name in {"siglip", "beit3", "od", "caption"}:
        marker["source_repo"] = KEYFRAME_REPO
    return marker


def test_quick_audit_reports_missing_module_files(tmp_path: Path) -> None:
    video_id = "L21_V001"
    files: dict[str, list[str]] = {}
    for contract in CONTRACTS:
        names = list(contract.required_files)
        if contract.name == "caption":
            names.remove("captions.parquet")
        files[contract.repo_id] = [f"data/L21/{video_id}/{name}" for name in names]
        if contract.marker_name in names:
            _write(
                tmp_path,
                contract.repo_id,
                f"data/L21/{video_id}/{contract.marker_name}",
                json.dumps(_marker(contract, video_id)),
            )

    frame, summary = audit_repositories(FakeHubClient(tmp_path, files), quick=True)

    assert summary["source_video_count"] == 1
    assert summary["audited_video_count"] == 1
    assert summary["error_video_count"] == 1
    assert frame.loc[0, "caption_status"] == "error"
    assert "captions.parquet" in frame.loc[0, "caption_errors"]
    assert frame.loc[0, "keyframe_status"] == "ok"
