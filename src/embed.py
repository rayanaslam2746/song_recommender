"""CLAP wrapper: raw audio samples -> 512-d embedding vector.

Does NOT normalize — normalization happens once in index_store, so there's a single
source of truth (SPEC.md §6, "Known pitfalls").
"""

import contextlib
import io
import os

import numpy as np
import laion_clap
import transformers

from config import CLAP_CKPT

# laion_clap's text tower init pulls in transformers (roberta-base), which by default
# logs a verbose "LOAD REPORT" at warning level on every model init. This is a one-time
# import-time setting, not tied to the CLAP_VERBOSE_LOAD toggle below.
transformers.logging.set_verbosity_error()

_model = None


def _verbose_load() -> bool:
    """CLAP_VERBOSE_LOAD=1 restores full checkpoint-loader output, e.g. when debugging
    a bad checkpoint load. Quiet (suppressed) by default.
    """
    return os.environ.get("CLAP_VERBOSE_LOAD", "").strip().lower() in ("1", "true", "yes")


@contextlib.contextmanager
def _quiet_unless_verbose():
    """Suppress stdout/stderr, unless CLAP_VERBOSE_LOAD asks to see it.

    laion_clap's load_ckpt() prints one "Loaded"/"Unloaded" line per model parameter
    (hundreds of lines); most of that is gated by its own `verbose` arg (used below),
    but a few surrounding lines (checkpoint path, "Load Checkpoint...", HF weight-loading
    progress bars) are unconditional print()/tqdm output, not logging, so lowering a log
    level won't catch them — the streams have to be redirected too.
    """
    if _verbose_load():
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _load_model():
    """Lazy-load the CLAP model once (module-level singleton). Wraps only this one-time
    init — never the ingest loop around it — so noisy checkpoint-loading output shows at
    most once per process, not once per track.
    """
    global _model
    if _model is not None:
        return _model

    ckpt_path = CLAP_CKPT if os.path.isabs(CLAP_CKPT) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CLAP_CKPT
    )

    verbose = _verbose_load()
    with _quiet_unless_verbose():
        try:
            # Locked choice (SPEC.md §3): music-pretrained checkpoint, noticeably better
            # musical matches than the default general-audio checkpoint.
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            model.load_ckpt(ckpt_path, verbose=verbose)
        except Exception:
            # Fallback (SPEC.md §3): if the music checkpoint fails to load, fall back to
            # the default general-audio checkpoint rather than hard-failing.
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt(verbose=verbose)

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
