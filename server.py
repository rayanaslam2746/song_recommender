"""FastAPI server for the web UI (WEB_APP_SPEC.md). Loads the CLAP model once at startup
so a query costs ~3-4s instead of the ~30s a fresh CLI process pays for the checkpoint
load.

catalog.parquet / index.faiss are cheap (sub-second) to re-read from disk, so — unlike
the model — request handlers just re-read them fresh each time via the existing
`recommend()` helper. That sidesteps an entire class of "in-memory cache went stale after
a cold-path append" bugs for a cost that's negligible next to the embedding step.
"""

import os
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    CATALOG_PATH,
    EMB_PATH,
    INDEX_PATH,
    ITUNES_SEARCH_URL,
    SEARCH_CACHE_MAX,
    SEARCH_CACHE_TTL,
    SEARCH_LIMIT,
    SERVER_HOST,
    SERVER_PORT,
)
from src import embed, ingest
from src.audio import load_clip
from src.embed import embed_samples
from src.index_store import load_index
from src.recommend import recommend

RECOMMEND_K = 10  # matches cli.py's `recommend --k` default

app = FastAPI()

# Guards the whole cold-path (embed + append + persist): two concurrent cold requests
# read-modify-write the same catalog/embeddings/index files, and without a lock they can
# interleave and corrupt row alignment (WEB_APP_SPEC.md §2 concurrency note).
_append_lock = threading.Lock()

# In-memory search cache, keyed by normalized query. FIFO eviction past SEARCH_CACHE_MAX
# keeps repeated/popular searches free of further iTunes calls (WEB_APP_SPEC.md §3).
_search_cache: "OrderedDict[str, tuple]" = OrderedDict()


def _cache_get(key: str):
    entry = _search_cache.get(key)
    if entry is None:
        return None
    expires_at, results = entry
    if time.time() > expires_at:
        del _search_cache[key]
        return None
    _search_cache.move_to_end(key)
    return results


def _cache_set(key: str, results: list) -> None:
    _search_cache[key] = (time.time() + SEARCH_CACHE_TTL, results)
    _search_cache.move_to_end(key)
    while len(_search_cache) > SEARCH_CACHE_MAX:
        _search_cache.popitem(last=False)


class RecommendRequest(BaseModel):
    itunes_track_id: Optional[int] = None
    title: str
    artist: str
    preview_url: Optional[str] = None


@app.on_event("startup")
def _startup():
    embed._load_model()  # the ~30s cost — pay it once, here, not per request
    if os.path.exists(INDEX_PATH):
        load_index(INDEX_PATH)  # touch once so a corrupt index fails fast at startup
    if os.path.exists(CATALOG_PATH):
        pd.read_parquet(CATALOG_PATH)
    print("ready")


def _load_catalog_for_lookup() -> pd.DataFrame:
    if not os.path.exists(CATALOG_PATH):
        return pd.DataFrame(columns=["track_id", "itunes_track_id"])
    return pd.read_parquet(CATALOG_PATH)


@app.get("/api/search")
def search(q: str):
    key = q.strip().lower()
    if not key:
        return {"results": []}

    cached = _cache_get(key)
    if cached is not None:
        return {"results": cached}

    try:
        resp = requests.get(
            ITUNES_SEARCH_URL,
            params={"term": q, "entity": "song", "limit": SEARCH_LIMIT},
            timeout=10,
        )
        resp.raise_for_status()
        raw_results = resp.json().get("results", [])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"iTunes search failed: {e}")

    catalog = _load_catalog_for_lookup()
    if "itunes_track_id" in catalog.columns:
        known_ids = set(catalog["itunes_track_id"].dropna().astype(int))
    else:
        known_ids = set()

    results = []
    for r in raw_results:
        track_id = r.get("trackId")
        if track_id is None:
            continue
        results.append(
            {
                "itunes_track_id": track_id,
                "title": r.get("trackName"),
                "artist": r.get("artistName"),
                "artwork": r.get("artworkUrl100"),
                "preview_url": r.get("previewUrl"),  # None is common — rights vary by region
                "in_catalog": track_id in known_ids,
            }
        )

    _cache_set(key, results)
    return {"results": results}


def _warm_lookup(itunes_track_id: int):
    """Return (query_vector, track_id) if itunes_track_id is already in the catalog,
    else None.
    """
    catalog = _load_catalog_for_lookup()
    if "itunes_track_id" not in catalog.columns:
        return None
    matches = catalog.index[catalog["itunes_track_id"] == itunes_track_id]
    if len(matches) == 0:
        return None
    row_pos = int(matches[0])
    embeddings = np.load(EMB_PATH)
    query_vector = embeddings[row_pos]
    track_id = int(catalog.iloc[row_pos]["track_id"])
    return query_vector, track_id


def _download_preview(url: str) -> str:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download preview: {e}")
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(resp.content)
        return tmp.name


def _cold_path(body: RecommendRequest) -> dict:
    tmp_path = _download_preview(body.preview_url)
    try:
        try:
            samples = load_clip(tmp_path)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"Failed to decode preview: {e}")
        query_vector = embed_samples(samples)
    finally:
        os.remove(tmp_path)

    if not np.isfinite(query_vector).all():
        # CLAP occasionally returns NaN/Inf for degenerate input (silence, a corrupted or
        # truncated preview). A NaN score would crash Starlette's JSON renderer (it sets
        # allow_nan=False) instead of returning a clean error, and appending it would
        # permanently poison every future search that happens to rank this track highly —
        # so catch it here, before either can happen.
        raise HTTPException(
            status_code=422,
            detail="Could not analyze this track's audio (looks like silence or a corrupted preview).",
        )

    # Search first, against the index as it exists before this track is added — matches
    # WEB_APP_SPEC.md §2's ordering and means no self-match to exclude.
    results = recommend(query_vector, k=RECOMMEND_K)

    ingest.append_track(
        query_vector,
        title=body.title,
        artist=body.artist,
        preview_url=body.preview_url,
        itunes_track_id=body.itunes_track_id,
    )

    return {"results": results, "was_cold": True}


@app.post("/api/recommend")
def recommend_endpoint(body: RecommendRequest):
    try:
        if body.itunes_track_id is not None:
            warm = _warm_lookup(body.itunes_track_id)
            if warm is not None:
                query_vector, track_id = warm
                results = recommend(query_vector, k=RECOMMEND_K, exclude_track_id=track_id)
                return {"results": results, "was_cold": False}

        if not body.preview_url:
            raise HTTPException(status_code=400, detail="This track has no preview available.")

        with _append_lock:
            return _cold_path(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


# Mounted last (and at "/") so the /api/* routes above take precedence — Starlette
# matches routes in registration order, and a root Mount would otherwise shadow them.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT)
