"""Per-branch search over an index built by ``retrieval.build``.

Every branch returns ``Hit`` objects keyed by the shared ``frame_row`` id, ranked best
first, so branches can be fused without comparing their incompatible raw scores.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multimedia_video_rag.retrieval.text import (
    ENGLISH_STOPWORDS,
    fts_or_query,
    normalize_search,
    remove_accents,
    tokens,
)


@dataclass(frozen=True)
class Hit:
    frame_row: int
    score: float
    evidence: str = ""


@dataclass(frozen=True)
class ObjectConstraint:
    label: str
    min_count: int = 1

    @classmethod
    def parse(cls, value: str) -> ObjectConstraint:
        """Parse ``"car"`` or ``"car:2"``."""
        label, _, count = value.strip().partition(":")
        if not label.strip():
            raise ValueError(f"Empty object label in {value!r}")
        return cls(label.strip().casefold(), int(count) if count else 1)


def parse_objects(value: str | None) -> list[ObjectConstraint]:
    """Parse ``"car:2; person"`` (``;`` or ``,`` separated)."""
    if not value:
        return []
    parts = [part for part in value.replace(",", ";").split(";") if part.strip()]
    return [ObjectConstraint.parse(part) for part in parts]


class RetrievalIndex:
    """Read-only access to one index directory (FAISS files are loaded on first use)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.db = sqlite3.connect(
            f"file:{(self.root / self.manifest['sqlite']).as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.frames = pd.read_sql(
            "SELECT frame_row, frame_uid, video_id, level, shot_id, timestamp_sec, image_path "
            "FROM frames ORDER BY frame_row",
            self.db,
        )
        if not self.frames["frame_row"].equals(pd.Series(range(len(self.frames)))):
            raise ValueError("frames.frame_row must be 0..N-1")
        self._faiss: dict[str, Any] = {}
        # Integer codes so temporal neighbours can be checked with vectorized numpy.
        self.video_codes = pd.factorize(self.frames["video_id"])[0]
        self.shot_ids = self.frames["shot_id"].to_numpy()
        self._video_times = {
            video_id: (
                group["timestamp_sec"].to_numpy(),
                group["frame_row"].to_numpy(),
            )
            for video_id, group in self.frames.sort_values(["video_id", "timestamp_sec"]).groupby(
                "video_id", sort=False
            )
        }

    # ---- visual -------------------------------------------------------------------------

    @property
    def visual_modules(self) -> list[str]:
        return list(self.manifest["faiss"])

    def faiss_index(self, module: str) -> Any:
        if module not in self._faiss:
            import faiss

            entry = self.manifest["faiss"][module]
            self._faiss[module] = faiss.read_index(str(self.root / entry["file"]))
        return self._faiss[module]

    def release_faiss(self) -> None:
        """Drop loaded FAISS indexes (reloaded on the next visual search)."""
        self._faiss.clear()

    def visual_batch(self, module: str, vectors: np.ndarray, k: int = 1000) -> list[list[Hit]]:
        """Search several unit-norm query vectors at once (much faster than one by one)."""
        index = self.faiss_index(module)
        queries = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if queries.shape[1] != index.d:
            raise ValueError(f"{module}: query dim {queries.shape[1]} != index dim {index.d}")
        scores, rows = index.search(queries, k)
        return [
            [Hit(int(row), float(score)) for row, score in zip(r, s, strict=True) if row >= 0]
            for r, s in zip(rows, scores, strict=True)
        ]

    def reconstruct(self, module: str, frame_rows: np.ndarray) -> np.ndarray:
        """Stored (FP16-decoded) image vectors for the given frame rows, as float32."""
        rows = np.ascontiguousarray(frame_rows, dtype=np.int64)
        if len(rows) == 0:
            return np.empty((0, self.faiss_index(module).d), dtype=np.float32)
        return self.faiss_index(module).reconstruct_batch(rows)

    def visual(self, module: str, vector: np.ndarray, k: int = 1000) -> list[Hit]:
        return self.visual_batch(module, vector, k)[0]

    # ---- caption (English BM25) ---------------------------------------------------------

    def caption(self, text_en: str, k: int = 500) -> list[Hit]:
        query = fts_or_query(tokens(text_en, stopwords=ENGLISH_STOPWORDS))
        if query is None:
            return []
        rows = self.db.execute(
            "SELECT c.frame_row, bm25(captions_fts), c.caption_en "
            "FROM captions_fts JOIN captions c ON c.frame_row = captions_fts.rowid "
            "WHERE captions_fts MATCH ? ORDER BY bm25(captions_fts) LIMIT ?",
            [query, k],
        ).fetchall()
        return [Hit(int(row), -float(bm25), caption) for row, bm25, caption in rows]

    # ---- ASR (Vietnamese) ---------------------------------------------------------------

    def asr_chunks(self, text_vi: str, k: int = 200, candidates: int = 1000) -> pd.DataFrame:
        """Rank speech chunks: exact accented phrase > accented token overlap > BM25.

        FTS matching is accent-insensitive (the index folds diacritics), so "bão lũ" also
        matches "báo", "bảo"...; the Python re-rank restores accent precision.
        """
        plain_query = fts_or_query(tokens(remove_accents(text_vi)))
        if plain_query is None:
            return pd.DataFrame(columns=["chunk_row", "video_id", "start_sec", "end_sec", "text"])
        rows = self.db.execute(
            "SELECT c.chunk_row, c.video_id, c.start_sec, c.end_sec, c.normalized_text, "
            "bm25(asr_chunks_fts) "
            "FROM asr_chunks_fts JOIN asr_chunks c ON c.chunk_row = asr_chunks_fts.rowid "
            "WHERE asr_chunks_fts MATCH ? ORDER BY bm25(asr_chunks_fts) LIMIT ?",
            [f"normalized_no_accent : ({plain_query})", candidates],
        ).fetchall()
        chunks = pd.DataFrame(
            rows, columns=["chunk_row", "video_id", "start_sec", "end_sec", "text", "bm25"]
        )
        accented = tokens(text_vi)
        phrase = " ".join(accented)
        chunks["exact_phrase"] = [phrase in text for text in chunks["text"]]
        chunks["accented_hits"] = [
            len(set(accented) & set(normalize_search(text).split())) for text in chunks["text"]
        ]
        chunks = chunks.sort_values(
            ["exact_phrase", "accented_hits", "bm25"], ascending=[False, False, True], kind="stable"
        )
        return chunks.head(k).reset_index(drop=True)

    def frames_in_interval(
        self, video_id: str, start_sec: float, end_sec: float, pad_sec: float = 1.0
    ) -> np.ndarray:
        """Keyframes of a video inside [start, end] (padded); nearest one if none falls inside."""
        times, rows = self._video_times.get(video_id, (np.array([]), np.array([], dtype=int)))
        if len(times) == 0:
            return rows
        left = np.searchsorted(times, start_sec - pad_sec, side="left")
        right = np.searchsorted(times, end_sec + pad_sec, side="right")
        if right > left:
            return rows[left:right]
        middle = (start_sec + end_sec) / 2
        return rows[[int(np.argmin(np.abs(times - middle)))]]

    def asr(self, text_vi: str, k: int = 200) -> list[Hit]:
        """Frames covered by the best speech chunks; frames inherit their chunk's rank."""
        hits: list[Hit] = []
        seen: set[int] = set()
        chunks = self.asr_chunks(text_vi, k=k)
        for rank, chunk in enumerate(chunks.itertuples(index=False)):
            snippet = chunk.text[:160]
            for row in self.frames_in_interval(chunk.video_id, chunk.start_sec, chunk.end_sec):
                if int(row) not in seen:
                    seen.add(int(row))
                    hits.append(Hit(int(row), -float(rank), snippet))
        return hits

    # ---- object detection ---------------------------------------------------------------

    def objects(self, constraints: Sequence[ObjectConstraint], k: int = 500) -> list[Hit]:
        """Soft object match: frames satisfying more constraints first, then confidence.

        Labels match ``label_en`` or ``label_vi`` case-insensitively; an unknown label
        simply contributes nothing (fail-open).
        """
        if not constraints:
            return []
        clauses, params = [], []
        for constraint in constraints:
            clauses.append("((lower(label_en) = ? OR lower(label_vi) = ?) AND object_count >= ?)")
            params += [constraint.label, constraint.label, constraint.min_count]
        rows = self.db.execute(
            "SELECT frame_row, COUNT(*) AS satisfied, SUM(max_confidence) AS confidence, "
            "GROUP_CONCAT(label_en || 'x' || object_count, ', ') "
            f"FROM od_frame_labels WHERE {' OR '.join(clauses)} "
            "GROUP BY frame_row ORDER BY satisfied DESC, confidence DESC LIMIT ?",
            [*params, k],
        ).fetchall()
        return [
            Hit(int(row), float(satisfied) + float(confidence) / 100, str(labels))
            for row, satisfied, confidence, labels in rows
        ]

    def known_labels(self) -> list[str]:
        return [
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT label_en FROM od_frame_labels ORDER BY label_en"
            )
        ]

    # ---- helpers ------------------------------------------------------------------------

    def describe(self, frame_rows: Iterable[int]) -> pd.DataFrame:
        rows = [int(row) for row in frame_rows]
        info = self.frames.iloc[rows].copy()
        if not rows:
            info["caption_en"] = []
            return info.reset_index(drop=True)
        captions = dict(
            self.db.execute(
                f"SELECT frame_row, caption_en FROM captions WHERE frame_row IN "
                f"({','.join('?' * len(rows))})",
                rows,
            ).fetchall()
        )
        info["caption_en"] = [captions.get(row, "") for row in rows]
        return info.reset_index(drop=True)
