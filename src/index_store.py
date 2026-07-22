"""Build / save / load the FAISS index. Normalization happens here (single source of truth,
per SPEC.md §6/§10) — callers should never call faiss.normalize_L2 themselves.
"""

import faiss
import numpy as np

from config import EMBED_DIM


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Idempotent, so re-normalizing already-unit vectors is safe."""
    normalized = np.array(embeddings, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(normalized)
    return normalized


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """embeddings: (N, 512) float32 -> flat cosine-similarity index (inner product on L2-normalized vectors)."""
    normalized = normalize_embeddings(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(normalized)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)
