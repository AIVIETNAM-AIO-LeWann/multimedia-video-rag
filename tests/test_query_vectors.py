"""Query vectors produced on Colab/Kaggle and consumed locally."""

from __future__ import annotations

import ast
import contextlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from multimedia_video_rag.retrieval import encoders
from multimedia_video_rag.retrieval.query_vectors import (
    FORMAT_VERSION,
    Query,
    align_to,
    load_query_vectors,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/retrieval/encode-query-vectors.ipynb"
MANIFEST = {
    "faiss": {
        "siglip": {"dimension": 4, "model_ids": ["m/siglip"], "model_revisions": ["rev-s"]},
        "beit3": {"dimension": 3, "model_ids": ["m/beit3"], "model_revisions": ["sha-b"]},
    }
}


def unit(rows: int, dimension: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(rows, dimension)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def write_vectors(path: Path, *, siglip_revision: str = "rev-s", scale: float = 1.0) -> Path:
    beit3 = unit(2, 3, 1)
    beit3[1] = np.nan  # second query has no English text
    meta = {
        "format_version": FORMAT_VERSION,
        "siglip_text": "vi",
        "models": {
            "siglip": {"model_id": "m/siglip", "model_revision": siglip_revision, "dimension": 4},
            "beit3": {"model_id": "m/beit3", "model_revision": "sha-b", "dimension": 3},
        },
    }
    np.savez(
        path,
        meta=np.array(json.dumps(meta)),
        query_id=np.array(["q1", "q2"]),
        query_vi=np.array(["bão lũ", "xe buýt"]),
        query_en=np.array(["flood", ""]),
        objects=np.array(["car:2", ""]),
        siglip=unit(2, 4, 0) * scale,
        beit3=beit3,
    )
    return path


def test_load_and_align(tmp_path: Path):
    loaded = load_query_vectors(write_vectors(tmp_path / "v.npz"), MANIFEST)
    assert [query.query_id for query in loaded.queries] == ["q1", "q2"]
    assert loaded.queries[0] == Query("q1", "bão lũ", "flood")  # old "objects" array ignored
    assert np.isnan(loaded.vectors["beit3"][1]).all()

    wanted = [Query("q2", "xe buýt", ""), Query("q1", "bão lũ", "flood")]
    aligned = align_to(loaded, wanted)
    np.testing.assert_array_equal(aligned["siglip"][0], loaded.vectors["siglip"][1])

    with pytest.raises(ValueError, match="changed since it was encoded"):
        align_to(loaded, [Query("q1", "bão lũ lớn", "flood")])
    with pytest.raises(KeyError):
        align_to(loaded, [Query("q9", "", "")])


def test_rejects_other_model_revision(tmp_path: Path):
    path = write_vectors(tmp_path / "v.npz", siglip_revision="other")
    with pytest.raises(ValueError, match="index images use"):
        load_query_vectors(path, MANIFEST)


def test_rejects_unnormalized_vectors(tmp_path: Path):
    with pytest.raises(ValueError, match="not L2-normalized"):
        load_query_vectors(write_vectors(tmp_path / "v.npz", scale=2.0), MANIFEST)


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def notebook_constants(source: str) -> dict[str, object]:
    constants = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            # Computed values (paths, runtime) are not literals and are skipped.
            if isinstance(target, ast.Name) and target.id.isupper():
                with contextlib.suppress(ValueError):
                    constants[target.id] = ast.literal_eval(node.value)
    return constants


def test_notebook_matches_local_encoders():
    source = notebook_source()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
    constants = notebook_constants(source)
    for name in (
        "BEIT3_SOURCE_REVISION",
        "BEIT3_SOURCE_SHA256",
        "BEIT3_CHECKPOINT_URL",
        "BEIT3_CHECKPOINT_SIZE",
        "TORCHSCALE_URL",
        "TORCHSCALE_SHA256",
        "BEIT3_TOKENIZER_URL",
        "SIGLIP_MAX_LENGTH",
        "BEIT3_MAX_LENGTH",
    ):
        assert constants[name] == getattr(encoders, name), name
    assert constants["FORMAT_VERSION"] == FORMAT_VERSION
    assert "str(text).lower()" in source  # SigLIP 2 lowercases
    assert 'padding="max_length"' in source
    assert '"transformers": "4.57.1"' in source and '"timm": "0.9.16"' in source


def test_notebook_beit3_inputs_equal_local_implementation():
    tree = ast.parse(notebook_source())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "beit3_text_inputs"
    )
    namespace = {"BEIT3_MAX_LENGTH": encoders.BEIT3_MAX_LENGTH}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook", "exec"), namespace)
    for ids in ([5, 6, 7], list(range(100, 200))):
        assert namespace["beit3_text_inputs"](ids, 0, 2, 1) == encoders.beit3_text_inputs(
            ids, bos=0, eos=2, pad=1
        )
