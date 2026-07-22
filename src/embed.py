"""CLAP wrapper: raw audio samples -> 512-d embedding vector.

Does NOT normalize — normalization happens once in index_store, so there's a single
source of truth (SPEC.md §6, "Known pitfalls").
"""

import os

import numpy as np
import laion_clap

from config import CLAP_CKPT

_model = None


def _load_model():
    """Lazy-load the CLAP model once (module-level singleton)."""
    global _model
    if _model is not None:
        return _model

    ckpt_path = CLAP_CKPT if os.path.isabs(CLAP_CKPT) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CLAP_CKPT
    )

    try:
        # Locked choice (SPEC.md §3): music-pretrained checkpoint, noticeably better
        # musical matches than the default general-audio checkpoint.
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(ckpt_path)
    except Exception:
        # Fallback (SPEC.md §3): if the music checkpoint fails to load, fall back to
        # the default general-audio checkpoint rather than hard-failing.
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

    model.eval()
    _model = model
    return _model


def embed_samples(samples: np.ndarray) -> np.ndarray:
    """samples: (T,) float32 mono waveform -> (512,) float32 embedding."""
    return embed_batch([samples])[0]


def embed_batch(list_of_samples) -> np.ndarray:
    """list of (T,) float32 waveforms -> (B, 512) float32 embeddings."""
    model = _load_model()
    batch = np.stack([np.asarray(s, dtype=np.float32) for s in list_of_samples])
    embedding = model.get_audio_embedding_from_data(x=batch, use_tensor=False)
    return np.asarray(embedding, dtype=np.float32)
