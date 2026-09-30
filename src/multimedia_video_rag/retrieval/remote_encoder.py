"""Client for notebooks/retrieval/query-encoder-server.ipynb (query text -> vectors).

The server runs the SigLIP 2 / BEiT-3 text encoders on Colab or Kaggle; every response
carries the model revisions, which must match the index manifest like a vectors file does.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any

import numpy as np

from multimedia_video_rag.retrieval.query_vectors import VISUAL_MODULES

SERVER_VERSION = 1


class RemoteEncoderError(RuntimeError):
    pass


def check_models(meta: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Refuse a server whose encoders differ from the ones used to embed the index images."""
    if meta.get("server_version") != SERVER_VERSION:
        raise RemoteEncoderError(f"Unsupported server_version {meta.get('server_version')}")
    for module in VISUAL_MODULES:
        entry = manifest["faiss"][module]
        served = meta["models"][module]
        if served["model_revision"] not in entry["model_revisions"]:
            raise RemoteEncoderError(
                f"{module}: server uses revision {served['model_revision']}, "
                f"index images use {entry['model_revisions']}"
            )
        if served["dimension"] != entry["dimension"]:
            raise RemoteEncoderError(f"{module}: dimension {served['dimension']}")


def parse_vectors(
    payload: dict[str, Any], manifest: dict[str, Any], count: int
) -> dict[str, np.ndarray]:
    check_models(payload, manifest)
    vectors = {}
    for module in VISUAL_MODULES:
        matrix = np.asarray(payload[module], dtype=np.float32)
        if matrix.shape != (count, manifest["faiss"][module]["dimension"]):
            raise RemoteEncoderError(f"{module}: unexpected shape {matrix.shape}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
            raise RemoteEncoderError(f"{module}: vectors are not L2-normalized")
        vectors[module] = matrix
    return vectors


class RemoteEncoder:
    def __init__(
        self,
        url: str,
        key: str,
        manifest: dict[str, Any],
        *,
        timeout: float = 60.0,
        cache_size: int = 512,
    ) -> None:
        self.url = url.rstrip("/")
        self.key = key
        self.manifest = manifest
        self.timeout = timeout
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = cache_size

    def _request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"X-API-Key": self.key, "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RemoteEncoderError(f"Encoder server HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RemoteEncoderError(f"Encoder server unreachable: {error}") from error

    def health(self) -> dict[str, Any]:
        meta = self._request("/health")
        check_models(meta, self.manifest)
        return meta

    def encode(self, text: str) -> dict[str, np.ndarray]:
        """Vectors of one query, keyed by module (1-D, L2-normalized)."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("Empty query")
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        payload = self._request("/encode", {"texts": [text]})
        vectors = {
            module: matrix[0] for module, matrix in parse_vectors(payload, self.manifest, 1).items()
        }
        self._cache[text] = vectors
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vectors
