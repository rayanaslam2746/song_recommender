"""Build catalog.parquet + embeddings.npy (+ index.faiss) from a folder of audio, or from
an iTunes CSV. Both entry paths append to the same catalog + embeddings, keeping
`catalog.parquet` row i, `embeddings.npy` row i, and FAISS vector i aligned (SPEC.md §5,
§10 "Row alignment is sacred").
"""

import os
import re
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config import AUDIO_DIR, CATALOG_PATH, EMB_PATH, EMBED_DIM, INDEX_PATH, ITUNES_SLEEP
from src.audio import load_clip
from src.embed import embed_batch
from src.index_store import build_index, normalize_embeddings, save_index

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".aiff", ".wma"}

CATALOG_DTYPES = {
    "track_id": "int64",
    "title": "object",
    "artist": "object",
    "source_path": "object",
    "preview_url": "object",
}


def _find_audio_files(audio_dir: str) -> list:
    files = []
    for root, _, names in os.walk(audio_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


def _sanitize_filename(name: str) -> str:
    """Windows forbids <>:"/\\|?* in filenames; iTunes track titles sometimes contain them."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _derive_title_artist(path: str):
    """Best-effort, display-only (SPEC.md §6). Expects "Artist - Title.ext"-style names;
    falls back to the filename as title with an unknown artist.
    """
    stem = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem, "Unknown"


def _load_catalog() -> pd.DataFrame:
    if os.path.exists(CATALOG_PATH):
        return pd.read_parquet(CATALOG_PATH)
    return pd.DataFrame(columns=list(CATALOG_DTYPES)).astype(CATALOG_DTYPES)


def _load_embeddings() -> np.ndarray:
    if os.path.exists(EMB_PATH):
        return np.load(EMB_PATH)
    return np.zeros((0, EMBED_DIM), dtype=np.float32)


def ingest_folder(audio_dir: str = AUDIO_DIR, preview_urls: dict = None) -> None:
    """Recursively scan `audio_dir`, embed any tracks not already in the catalog, and
    rebuild catalog.parquet / embeddings.npy / index.faiss. Idempotent: re-running with
    no new files leaves the catalog unchanged.

    `preview_urls`, if given, maps source_path -> iTunes previewUrl for any of the new
    files (used by `ingest_itunes`); files not in the map get `preview_url=None`.
    """
    preview_urls = preview_urls or {}
    catalog = _load_catalog()
    embeddings = _load_embeddings()
    existing_paths = set(catalog["source_path"])

    candidates = [p for p in _find_audio_files(audio_dir) if p not in existing_paths]

    new_rows = []
    new_samples = []
    for path in tqdm(candidates, desc="Embedding audio"):
        try:
            samples = load_clip(path)
        except RuntimeError as e:
            # Decode failure: log and skip, don't crash the batch (SPEC.md §6).
            print(f"[skip] {e}")
            continue
        title, artist = _derive_title_artist(path)
        new_rows.append(
            {"title": title, "artist": artist, "source_path": path, "preview_url": preview_urls.get(path)}
        )
        new_samples.append(samples)

    if new_samples:
        new_embeddings = embed_batch(new_samples)
        embeddings = np.concatenate([embeddings, new_embeddings], axis=0)
        new_df = pd.DataFrame(new_rows)
        new_df["track_id"] = 0  # placeholder, reassigned below from row index
        catalog = pd.concat([catalog, new_df[list(CATALOG_DTYPES)]], ignore_index=True)
        catalog["track_id"] = catalog.index  # track_id = stable row index (SPEC.md §5)

    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)

    # Normalize once here (index_store is the single source of truth) so embeddings.npy
    # and the FAISS index always agree (SPEC.md §5, §10).
    normalized = normalize_embeddings(embeddings)
    np.save(EMB_PATH, normalized)
    catalog.to_parquet(CATALOG_PATH, index=False)

    index = build_index(normalized)
    save_index(index, INDEX_PATH)

    print(
        f"catalog: {len(catalog)} rows | embeddings: {normalized.shape} | "
        f"index: {index.ntotal} vectors | added {len(new_samples)} new track(s)"
    )


def ingest_itunes(csv_path: str) -> None:
    """Read a CSV of artist,title rows, fetch each track's ~30s iTunes preview into
    AUDIO_DIR, then embed via the same pipeline as ingest_folder. Skips rows whose
    preview file already exists (idempotent-ish, SPEC.md §6) and rows with no preview
    available, logging both rather than crashing the batch.
    """
    rows = pd.read_csv(csv_path)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    preview_urls = {}
    for _, row in rows.iterrows():
        artist = str(row["artist"]).strip()
        title = str(row["title"]).strip()

        safe_name = _sanitize_filename(f"{artist} - {title}")
        # Any extension would do for the existence check; iTunes previews are .m4a.
        out_path = os.path.join(AUDIO_DIR, f"{safe_name}.m4a")
        if os.path.exists(out_path):
            print(f"[skip] {artist} - {title}: preview already downloaded")
            continue

        try:
            resp = requests.get(
                "https://itunes.apple.com/search",
                params={"term": f"{artist} {title}", "entity": "song", "limit": 1},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.RequestException as e:
            print(f"[skip] {artist} - {title}: iTunes search failed ({e})")
            continue

        if not results or not results[0].get("previewUrl"):
            print(f"[skip] {artist} - {title}: no iTunes preview found")
            continue

        preview_url = results[0]["previewUrl"]
        try:
            audio_resp = requests.get(preview_url, timeout=30)
            audio_resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[skip] {artist} - {title}: preview download failed ({e})")
            continue

        with open(out_path, "wb") as f:
            f.write(audio_resp.content)
        preview_urls[out_path] = preview_url
        print(f"[ok] {artist} - {title} -> {out_path}")

        time.sleep(ITUNES_SLEEP)  # iTunes rate-limits around 20/min (SPEC.md §6)

    if preview_urls:
        ingest_folder(AUDIO_DIR, preview_urls=preview_urls)
    else:
        print("No new previews downloaded.")
