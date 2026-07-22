"""Smoke test (SPEC.md §11): embed 2 clearly-similar clips and 1 clearly-different clip,
assert the similar pair scores higher than either does against the outlier.

Uses synthesized signals (not files under data/audio/, which are gitignored and not
guaranteed to exist) so this runs offline and reproducibly for anyone who clones the repo.
"""

import numpy as np

from config import CLIP_SECONDS, SAMPLE_RATE
from src.embed import embed_samples
from src.index_store import normalize_embeddings


def _tone(freq: float) -> np.ndarray:
    """A rhythmic tone at `freq` Hz -- reads as more "musical" than a bare sine wave."""
    t = np.linspace(0, CLIP_SECONDS, SAMPLE_RATE * CLIP_SECONDS, endpoint=False)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    return (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, SAMPLE_RATE * CLIP_SECONDS).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embeddings(a.reshape(1, -1))
    b = normalize_embeddings(b.reshape(1, -1))
    return float(np.dot(a[0], b[0]))


def test_similar_pair_scores_higher_than_outlier():
    clip_a = _tone(440.00)  # A4
    clip_b = _tone(466.16)  # A#4 -- close in pitch, same rhythmic envelope as clip_a
    clip_c = _noise(seed=42)  # broadband noise -- clearly different from both

    emb_a = embed_samples(clip_a)
    emb_b = embed_samples(clip_b)
    emb_c = embed_samples(clip_c)

    sim_similar_pair = _cosine(emb_a, emb_b)
    sim_a_outlier = _cosine(emb_a, emb_c)
    sim_b_outlier = _cosine(emb_b, emb_c)

    assert sim_similar_pair > sim_a_outlier
    assert sim_similar_pair > sim_b_outlier
