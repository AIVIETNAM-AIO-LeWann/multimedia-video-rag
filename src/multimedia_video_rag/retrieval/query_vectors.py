"""Query vectors encoded elsewhere (Colab/Kaggle) and searched locally.

File format (``.npz``), written by ``notebooks/retrieval/encode-query-vectors.ipynb``:

- ``query_id``, ``query_vi``, ``query_en``, ``objects``: string arrays, one per query;
- ``siglip`` (N x 1152) and ``beit3`` (N x 1024) float32, L2-normalized; a row of NaN
  means the query had no text for that encoder (BEiT-3 needs ``query_en``);
- ``meta``: JSON with ``format_version``, the text language used for SigLIP 2, and per
  module ``model_id`` / ``model_revision``, which must match the index manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from multimedia_video_rag.retrieval.pipeline import Query
from multimedia_video_rag.retrieval.search import parse_objects

FORMAT_VERSION = 1
VISUAL_MODULES = ("siglip", "beit3")


@dataclass
class QueryVectors:
    queries: list[Query]
    vectors: dict[str, np.ndarray]
    meta: dict[str, Any]

    def row(self, query_id: str) -> int:
        for position, query in enumerate(self.queries):
            if query.query_id == query_id:
                return position
        raise KeyError(f"Query {query_id!r} not in vectors file")


def load_query_vectors(path: Path, manifest: dict[str, Any]) -> QueryVectors:
    """Load and check the file against the index: same encoder revisions and dimensions."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format_version {meta.get('format_version')}")
        ids = [str(value) for value in data["query_id"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate query_id")
        queries = [
            Query(
                query_id=query_id,
                text_vi=str(text_vi),
                text_en=str(text_en),
                objects=tuple(parse_objects(str(objects))),
            )
            for query_id, text_vi, text_en, objects in zip(
                ids, data["query_vi"], data["query_en"], data["objects"], strict=True
            )
        ]
        vectors: dict[str, np.ndarray] = {}
        for module in VISUAL_MODULES:
            if module not in data.files:
                continue
            entry = manifest["faiss"][module]
            encoded_with = meta["models"][module]
            if encoded_with["model_revision"] not in entry["model_revisions"]:
                raise ValueError(
                    f"{path}: {module} encoded with revision {encoded_with['model_revision']}, "
                    f"index images use {entry['model_revisions']}"
                )
            matrix = np.asarray(data[module], dtype=np.float32)
            if matrix.shape != (len(ids), entry["dimension"]):
                raise ValueError(f"{path}: {module} has shape {matrix.shape}")
            usable = ~np.isnan(matrix).any(axis=1)
            norms = np.linalg.norm(matrix[usable], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(f"{path}: {module} vectors are not L2-normalized")
            vectors[module] = matrix
    return QueryVectors(queries, vectors, meta)


def align_to(loaded: QueryVectors, queries: list[Query]) -> dict[str, np.ndarray]:
    """Reorder vectors to ``queries``, refusing texts edited after encoding."""
    positions = []
    for query in queries:
        position = loaded.row(query.query_id)
        encoded = loaded.queries[position]
        if (encoded.text_vi, encoded.text_en) != (query.text_vi, query.text_en):
            raise ValueError(
                f"Query {query.query_id!r} changed since it was encoded; re-run the notebook"
            )
        positions.append(position)
    return {module: matrix[positions] for module, matrix in loaded.vectors.items()}
