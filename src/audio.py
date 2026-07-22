"""Load an audio file into a mono, 48kHz, centered 30s clip ready for CLAP."""

import numpy as np
import librosa

from config import SAMPLE_RATE, CLIP_SECONDS


def load_clip(path: str) -> np.ndarray:
    """Load `path`, resample to mono 48kHz, and return a centered 30s window.

    If the file is shorter than 30s, the whole clip is returned as-is.
    Raises a clear error on decode failure so callers can log-and-skip in batch jobs.
    """
    try:
        # sr=SAMPLE_RATE forces librosa to resample; never trust the source rate.
        samples, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio file: {path} ({e})") from e

    samples = samples.astype(np.float32)

    clip_len = SAMPLE_RATE * CLIP_SECONDS
    if len(samples) > clip_len:
        start = (len(samples) - clip_len) // 2
        samples = samples[start : start + clip_len]

    return samples
