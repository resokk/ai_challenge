"""Local BAAI/bge-m3 dense embeddings (sentence-transformers). Used only for embedding/search.

Chunks are packed at unit boundaries (paragraphs / lines) by the model tokenizer. An oversized
unit is split explicitly by token offsets and reported; nothing is silently truncated —
`embed` refuses inputs longer than the model's max_seq_length.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from camunda_rag.config import EmbeddingSettings

log = logging.getLogger(__name__)


def select_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device=mps requested but MPS is not available")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Chunk:
    text: str
    token_count: int
    split_note: str | None = None


class Embedder:
    """BGE-M3 encoder. With load_model=False only the tokenizer is loaded (chunking without vectors)."""

    def __init__(self, cfg: EmbeddingSettings, load_model: bool = True) -> None:
        if cfg.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # long-input warnings are expected: counting happens before chunking
        self.cfg = cfg
        source = str(cfg.model_path) if cfg.model_path else cfg.model_name
        local_only = cfg.offline or cfg.model_path is not None
        self.name = cfg.model_name if cfg.model_path is None else f"{cfg.model_name}@{cfg.model_path}"
        self.model = None
        if load_model:
            from sentence_transformers import SentenceTransformer

            self.device = select_device(cfg.device)
            self.model = SentenceTransformer(source, device=self.device, local_files_only=local_only)
            self.model.max_seq_length = max(cfg.max_tokens, 512)
            self.tokenizer = self.model.tokenizer
            get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
            self.dim = int(get_dim())
            log.info("embedding model %s on %s (dim=%d)", source, self.device, self.dim)
        else:
            from transformers import AutoTokenizer

            self.device = "none"
            self.dim = 0
            self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def chunk(self, units: list[str], reserve: int = 0) -> list[Chunk]:
        """Pack text units into chunks of <= max_tokens - reserve; split only units that alone exceed it."""
        budget = self.cfg.max_tokens - reserve
        out: list[Chunk] = []
        cur: list[str] = []
        cur_tokens = 0
        for unit in units:
            n = self.count(unit)
            if n > budget:
                if cur:
                    out.append(Chunk("\n".join(cur), self.count("\n".join(cur))))
                    cur, cur_tokens = [], 0
                out.extend(self._split_unit(unit, budget))
                continue
            if cur and cur_tokens + n > budget:
                out.append(Chunk("\n".join(cur), self.count("\n".join(cur))))
                cur, cur_tokens = [], 0
            cur.append(unit)
            cur_tokens += n
        if cur:
            out.append(Chunk("\n".join(cur), self.count("\n".join(cur))))
        return out

    def _split_unit(self, unit: str, budget: int) -> list[Chunk]:
        enc = self.tokenizer(unit, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = enc["offset_mapping"]
        step = budget - 2  # room for special tokens
        parts = []
        for i in range(0, len(offsets), step):
            window = offsets[i:i + step]
            text = unit[window[0][0]:window[-1][1]]
            parts.append(Chunk(text, len(window) + 2, split_note=f"oversized unit split by tokens ({len(offsets)} tokens)"))
        return parts

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("embedder was created without the model (load_model=False)")
        limit = self.model.max_seq_length
        for t in texts:
            if self.count(t) > limit:
                raise ValueError(f"text of {self.count(t)} tokens exceeds max_seq_length {limit}; chunk it first")
        vecs = self.model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return np.asarray(vecs, dtype="float32")
