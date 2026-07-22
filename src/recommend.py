"""Nearest-neighbor query, purely merit-ranked by embedding similarity.

Similarity comes ONLY from the FAISS search over embeddings. Artist/title are read from
the catalog strictly after the search returns, purely for display — they never feed the
vector search or the score, and results are not capped or filtered by artist. If two
songs from the same artist are the closest matches, both show up.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import requests

from config import CATALOG_PATH, EMB_PATH, INDEX_PATH
from src.audio import load_clip
from src.embed import embed_samples
from src.index_store import load_index, normalize_embeddings


def recommend(query_vector: np.ndarray, k: int = 10, exclude_track_id=None) -> list:
    index = load_index(INDEX_PATH)
    catalog = pd.read_parquet(CATALOG_PATH)

    if index.ntotal == 0:
        return []

    query = normalize_embeddings(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    # +1 covers the query track itself when excluded, so k results still come back.
    fetch_n = min(k + 1 if exclude_track_id is not None else k, index.ntotal)
    scores, ids = index.search(query, fetch_n)

    results = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        row = catalog.iloc[int(row_id)]
        track_id = int(row["track_id"])

        if exclude_track_id is not None and track_id == exclude_track_id:
            continue

        results.append(
            {
                "track_id": track_id,
                "title": row["title"],
                "artist": row["artist"],
                "score": float(score),
                "preview_url": row["preview_url"],
            }
        )
        if len(results) >= k:
            break

    return results


def recommend_from_file(path: str, k: int = 10) -> list:
    samples = load_clip(path)
    vector = embed_samples(samples)
    return recommend(vector, k=k)


def recommend_from_search(query_str: str, k: int = 10) -> list:
    """Look up `query_str` on iTunes, embed its ~30s preview on the fly (not necessarily
    already in the catalog), and run `recommend`.
    """
    resp = requests.get(
        "https://itunes.apple.com/search",
        params={"term": query_str, "entity": "song", "limit": 1},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results or not results[0].get("previewUrl"):
        raise RuntimeError(f"No iTunes preview found for search: {query_str!r}")

    preview_url = results[0]["previewUrl"]
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio_resp = requests.get(preview_url, timeout=30)
        audio_resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(audio_resp.content)
        return recommend_from_file(tmp_path, k=k)
    finally:
        os.remove(tmp_path)
