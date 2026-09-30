"""Query-time text encoders that share the embedding space of the ingested images.

Both encoders reproduce the ingest setup exactly: the same model revision / checkpoint
(taken from the index manifest) and the same text preprocessing as the model's training.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODELS_DIR = Path("models")

# SigLIP 2 was trained on lowercased text padded to exactly 64 tokens.
SIGLIP_MAX_LENGTH = 64

# BEiT-3: pinned exactly as in notebooks/ingestion/ingest-visual-beit3-large-coco-retrieval.
BEIT3_SOURCE_REVISION = "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5"
BEIT3_SOURCE_SHA256 = {
    "modeling_utils.py": "cc04b762b2c0ba32eb82d65a5e543adada156ce3584a351b56c87eb3de293a27",
    "modeling_finetune.py": "dfa9a2dbba7e7b2a46023da0d693f6e781d946d86b2b7a52e69ed4bd463d763f",
}
BEIT3_CHECKPOINT_URL = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_large_patch16_384_coco_retrieval.pth"
)
BEIT3_CHECKPOINT_SIZE = 1_350_590_595
TORCHSCALE_URL = (
    "https://files.pythonhosted.org/packages/d5/7f/"
    "2a20aeaf3264eb41750b882fc56a67d907042dcf3f84745331ee4e6e7636/torchscale-0.2.0.tar.gz"
)
TORCHSCALE_SHA256 = "ca0a3982d799d6b91320a538a6cc17ea9ccb6370231934805fae78f318fe4d2f"
BEIT3_TOKENIZER_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
# BEiT-3 COCO retrieval fine-tuning uses num_max_bpe_tokens=64 on raw (cased) captions.
BEIT3_MAX_LENGTH = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, expected_size: int | None = None) -> Path:
    """Resumable download with an atomic rename."""
    import requests

    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"Downloading {url} ...", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with partial.open(mode) as stream:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                stream.write(block)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"{url}: got {partial.stat().st_size} bytes, expected {expected_size}")
    partial.replace(destination)
    return destination


def _verified(path: Path, expected_sha256: str) -> Path:
    """Check a file's SHA-256 once; a stamp file avoids re-hashing large checkpoints."""
    stamp = path.with_suffix(path.suffix + ".sha256")
    signature = f"{expected_sha256} {path.stat().st_size} {path.stat().st_mtime_ns}"
    if stamp.exists() and stamp.read_text(encoding="utf-8") == signature:
        return path
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{path.name}: SHA-256 {actual} != expected {expected_sha256}")
    stamp.write_text(signature, encoding="utf-8")
    return path


def _torch_dtype(name: str) -> Any:
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


class SiglipTextEncoder:
    """Text tower of ``google/siglip2-so400m-patch16-384`` (vision weights are not loaded)."""

    def __init__(self, model_id: str, revision: str, *, dtype: str = "float32") -> None:
        import torch
        from transformers import AutoTokenizer, Siglip2TextModel

        self.model_id, self.revision = model_id, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = Siglip2TextModel.from_pretrained(
            model_id, revision=revision, torch_dtype=_torch_dtype(dtype)
        ).eval()
        torch.set_grad_enabled(False)

    @staticmethod
    def preprocess(text: str) -> str:
        return " ".join(str(text).lower().split())

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            [self.preprocess(text) for text in texts],
            padding="max_length",
            max_length=SIGLIP_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            features = self.model(input_ids=inputs["input_ids"]).pooler_output
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.cpu().numpy()


def load_encoder(
    manifest: dict[str, Any],
    module: str,
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    dtype: str = "float32",
) -> SiglipTextEncoder | Beit3TextEncoder:
    """Build the text encoder matching an index's visual module (revision from the manifest)."""
    entry = manifest["faiss"][module]
    revisions = entry.get("model_revisions") or []
    if len(revisions) != 1:
        raise ValueError(f"{module}: expected exactly one ingest revision, got {revisions}")
    if module == "siglip":
        return SiglipTextEncoder(entry["model_ids"][0], revisions[0], dtype=dtype)
    if module == "beit3":
        return Beit3TextEncoder(revisions[0], models_dir=models_dir, dtype=dtype)
    raise ValueError(f"No text encoder for visual module {module!r}")


def beit3_text_inputs(
    token_ids: list[int], *, bos: int, eos: int, pad: int, max_length: int = BEIT3_MAX_LENGTH
) -> tuple[list[int], list[int]]:
    """Mirror ``BaseDataset._get_text_segment`` of UniLM BEiT-3: bos + ids + eos, then pad.

    The padding mask is 1 on padded positions, as BEiT-3 expects.
    """
    if not token_ids:
        raise ValueError("BEiT-3 query must contain at least one token")
    tokens = [bos, *token_ids[: max_length - 2], eos]
    padding = max_length - len(tokens)
    return tokens + [pad] * padding, [0] * len(tokens) + [1] * padding


class Beit3TextEncoder:
    """Language branch of BEiT-3 Large COCO retrieval, verified against the ingest checkpoint."""

    def __init__(
        self,
        checkpoint_sha256: str,
        *,
        models_dir: Path = DEFAULT_MODELS_DIR,
        dtype: str = "float32",
    ) -> None:
        import torch
        from transformers import XLMRobertaTokenizer

        root = Path(models_dir) / "beit3"
        source_dir = root / f"unilm-{BEIT3_SOURCE_REVISION[:12]}"
        for filename, expected in BEIT3_SOURCE_SHA256.items():
            url = (
                "https://raw.githubusercontent.com/microsoft/unilm/"
                f"{BEIT3_SOURCE_REVISION}/beit3/{filename}"
            )
            _verified(_download(url, source_dir / filename), expected)
        checkpoint = _verified(
            _download(
                BEIT3_CHECKPOINT_URL,
                root / "beit3_large_patch16_384_coco_retrieval.pth",
                expected_size=BEIT3_CHECKPOINT_SIZE,
            ),
            checkpoint_sha256,
        )
        torchscale_root = self._prepare_torchscale(root)
        spm = _download(BEIT3_TOKENIZER_URL, root / "beit3.spm")

        model_factory = self._import_model_factory(source_dir, torchscale_root)
        model = model_factory(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=True)
        del state
        self.model = model.to(dtype=_torch_dtype(dtype)).eval()
        self.tokenizer = XLMRobertaTokenizer(str(spm))
        self.checkpoint_sha256 = checkpoint_sha256
        torch.set_grad_enabled(False)

    @staticmethod
    def _prepare_torchscale(root: Path) -> Path:
        archive = _verified(
            _download(TORCHSCALE_URL, root / "torchscale-0.2.0.tar.gz"), TORCHSCALE_SHA256
        )
        target = root / "torchscale-0.2.0"
        if not (target / "torchscale" / "__init__.py").exists():
            with tarfile.open(archive, mode="r:gz") as tar:
                tar.extractall(root, filter="data")
        return target

    @staticmethod
    def _import_model_factory(source_dir: Path, torchscale_root: Path) -> Any:
        """Import UniLM's model code with the same inference-only shims as the ingest notebook."""
        import torch.nn as nn

        fairscale = types.ModuleType("fairscale")
        fairscale_nn = types.ModuleType("fairscale.nn")
        fairscale_nn.checkpoint_wrapper = lambda module, *args, **kwargs: module
        fairscale_nn.wrap = lambda module, *args, **kwargs: module
        fairscale.nn = fairscale_nn

        class InferenceOnlyClipLoss(nn.Module):
            def forward(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Training loss is unavailable at query time")

        utils_shim = types.ModuleType("utils")
        utils_shim.ClipLoss = lambda *args, **kwargs: InferenceOnlyClipLoss()
        utils_shim.get_rank = lambda: 0
        utils_shim.get_world_size = lambda: 1

        shims = {"fairscale": fairscale, "fairscale.nn": fairscale_nn, "utils": utils_shim}
        saved = {name: sys.modules.get(name) for name in shims}
        sys.modules.update(shims)
        sys.path[:0] = [str(torchscale_root), str(source_dir)]
        try:
            from modeling_finetune import beit3_large_patch16_384_retrieval
        finally:
            # Keep the global "utils" name free for other packages once the code is imported.
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return beit3_large_patch16_384_retrieval

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        tokenizer = self.tokenizer
        batch_tokens, batch_padding = [], []
        for text in texts:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(str(text).split())))
            tokens, padding = beit3_text_inputs(
                ids,
                bos=tokenizer.bos_token_id,
                eos=tokenizer.eos_token_id,
                pad=tokenizer.pad_token_id,
            )
            batch_tokens.append(tokens)
            batch_padding.append(padding)
        with torch.inference_mode():
            _, language = self.model(
                text_description=torch.tensor(batch_tokens),
                padding_mask=torch.tensor(batch_padding),
                only_infer=True,
            )
        features = torch.nn.functional.normalize(language.float(), dim=-1)
        return features.cpu().numpy()
