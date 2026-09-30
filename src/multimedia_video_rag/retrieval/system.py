"""Baseline interactive search: SigLIP 2 + BEiT-3 keyframe retrieval with a chosen fusion.

Experimental starting point, not a settled design. Each encoder contributes its exact top
``pool_k`` keyframes; every candidate in the union is then scored by *both* encoders (dot
product with the stored vectors), so fusion never has to guess a missing score:

- ``rrf``: w / (60 + rank_siglip) + (1 - w) / (60 + rank_beit3), ranks within the pool;
- ``max-norm``: w * s_siglip / max s_siglip + (1 - w) * s_beit3 / max s_beit3
  (the ensemble of arXiv:2504.08384, Algorithm 3);
- ``siglip`` / ``beit3``: one encoder alone.

``collapse_shots`` keeps the best keyframe per (video, shot) so one shot cannot fill the page.

With a ``DenseScorer`` the whole corpus is scored by a matrix-vector product over FP16
copies of the index vectors (torch when installed, ~0.1 s per encoder on CPU); without one,
the FAISS indexes are searched. Both give the same candidates and scores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multimedia_video_rag.retrieval.fusion import RRF_K
from multimedia_video_rag.retrieval.search import RetrievalIndex

MODULES = ("siglip", "beit3")
FUSIONS = ("rrf", "max-norm", "siglip", "beit3")


class DenseScorer:
    """Exact full-corpus scores from FP16 copies of the stored (FP16) index vectors.

    The matrices are decoded from the FAISS indexes once and cached as ``.npy`` under
    ``cache_dir`` (outside the index directory), keyed by the index build time.
    """

    def __init__(self, index: RetrievalIndex, cache_dir: Path, *, chunk: int = 50_000) -> None:
        self.matrices: dict[str, np.ndarray] = {}
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = str(index.manifest.get("created_at", ""))
        for module in MODULES:
            path = cache_dir / f"{module}.fp16.npy"
            meta_path = cache_dir / f"{module}.fp16.json"
            faiss_index = index.faiss_index(module)
            shape = (faiss_index.ntotal, faiss_index.d)
            meta = {"created_at": stamp, "shape": list(shape)}
            cached = meta_path.is_file() and path.is_file()
            if cached and json.loads(meta_path.read_text(encoding="utf-8")) == meta:
                self.matrices[module] = np.load(path)
                continue
            matrix = np.empty(shape, dtype=np.float16)
            for start in range(0, shape[0], chunk):
                count = min(chunk, shape[0] - start)
                matrix[start : start + count] = faiss_index.reconstruct_n(start, count)
            partial = path.with_suffix(".partial.npy")
            np.save(partial, matrix)
            partial.replace(path)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            self.matrices[module] = matrix
        index.release_faiss()
        try:
            import torch
        except ImportError:
            self._tensors = None
        else:
            self._tensors = {m: torch.from_numpy(x) for m, x in self.matrices.items()}

    def scores(self, module: str, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._tensors is not None:
            import torch

            with torch.inference_mode():
                vector = torch.from_numpy(query.astype(np.float16))
                return (self._tensors[module] @ vector).float().numpy()
        matrix = self.matrices[module]
        out = np.empty(len(matrix), dtype=np.float32)
        for start in range(0, len(matrix), 16_384):
            out[start : start + 16_384] = matrix[start : start + 16_384].astype(np.float32) @ query
        return out


@dataclass(frozen=True)
class SearchResult:
    frame_row: int
    score: float
    similarity: dict[str, float]
    rank: dict[str, int]


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks, highest score first; ties keep candidate order."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def dual_encoder_search(
    index: RetrievalIndex,
    vectors: Mapping[str, np.ndarray],
    *,
    fusion: str = "rrf",
    weight_siglip: float = 0.5,
    pool_k: int = 1000,
    collapse_shots: bool = True,
    limit: int = 200,
    scorer: DenseScorer | None = None,
) -> list[SearchResult]:
    if fusion not in FUSIONS:
        raise ValueError(f"Unknown fusion {fusion!r}; choose from {FUSIONS}")
    if not 0.0 <= weight_siglip <= 1.0:
        raise ValueError("weight_siglip must be in [0, 1]")
    queries = {
        module: np.asarray(vectors[module], dtype=np.float32).reshape(-1) for module in MODULES
    }
    if scorer is not None:
        full = {module: scorer.scores(module, queries[module]) for module in MODULES}
        k = min(pool_k, len(full[MODULES[0]]))
        pool = np.unique(np.concatenate([np.argpartition(-full[m], k - 1)[:k] for m in MODULES]))
        rows = pool.astype(np.int64)
        sims = {module: full[module][rows] for module in MODULES}
    else:
        found: set[int] = set()
        for module in MODULES:
            found.update(hit.frame_row for hit in index.visual(module, queries[module], k=pool_k))
        rows = np.array(sorted(found), dtype=np.int64)
        if len(rows) == 0:
            return []
        sims = {module: index.reconstruct(module, rows) @ queries[module] for module in MODULES}
    ranks = {module: _ranks(sims[module]) for module in MODULES}

    weights = {"siglip": weight_siglip, "beit3": 1.0 - weight_siglip}
    if fusion == "rrf":
        score = sum(weights[m] / (RRF_K + ranks[m]) for m in MODULES)
    elif fusion == "max-norm":
        score = sum(weights[m] * sims[m] / max(float(sims[m].max()), 1e-6) for m in MODULES)
    else:
        score = sims[fusion]

    order = np.lexsort((rows, -score))
    videos = index.video_codes[rows]
    shots = index.shot_ids[rows]
    results, seen = [], set()
    for i in order:
        if collapse_shots:
            key = (int(videos[i]), int(shots[i]))
            if key in seen:
                continue
            seen.add(key)
        results.append(
            SearchResult(
                frame_row=int(rows[i]),
                score=float(score[i]),
                similarity={m: float(sims[m][i]) for m in MODULES},
                rank={m: int(ranks[m][i]) for m in MODULES},
            )
        )
        if len(results) == limit:
            break
    return results


def video_context(index: RetrievalIndex, frame_row: int, radius: int = 8) -> np.ndarray:
    """Frame rows of the same video within ``radius`` keyframes (rows are video-contiguous)."""
    codes = index.video_codes
    lo = max(0, frame_row - radius)
    hi = min(len(codes), frame_row + radius + 1)
    rows = np.arange(lo, hi)
    return rows[codes[rows] == codes[frame_row]]
