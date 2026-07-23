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

from config import (
    AUDIO_DIR,
    CATALOG_PATH,
    EMB_PATH,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    INDEX_PATH,
    ITUNES_CHECKPOINT_EVERY,
    ITUNES_SLEEP,
)
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
    # Nullable: only known for tracks fetched via ingest_itunes/append_track. Used for
    # exact "do we already have this song?" lookups (WEB_APP_SPEC.md §1) instead of fuzzy
    # artist/title string matching.
    "itunes_track_id": "Int64",
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


def _persist(catalog: pd.DataFrame, embeddings: np.ndarray):
    """Normalize + save catalog.parquet, embeddings.npy, and rebuild+save index.faiss so
    all three always agree, even if the process is killed right after this call returns
    (SPEC.md §5, §10 "row alignment is sacred"). Cheap even at thousands of rows, so it's
    safe to call after every chunk, not just once at the end.

    Each file is written to a `.tmp` sibling and atomically swapped in via os.replace(),
    rather than written in place — the web app (WEB_APP_SPEC.md) reads these same files
    from request handlers with no lock of their own, so a reader landing mid-`np.save()`
    would otherwise get a truncated array that silently decodes to garbage floats
    (NaN/Inf), not an error. os.replace() is atomic on both Windows and POSIX, so a
    concurrent reader always sees either the fully-old or fully-new file, never a torn one.
    """
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    normalized = normalize_embeddings(embeddings)

    emb_tmp = EMB_PATH + ".tmp"
    np.save(emb_tmp, normalized)
    # np.save appends .npy if the name doesn't already end with it — EMB_PATH already
    # does, so the tmp name picks up a second .npy (EMB_PATH + ".tmp.npy"); locate it.
    if not os.path.exists(emb_tmp) and os.path.exists(emb_tmp + ".npy"):
        emb_tmp = emb_tmp + ".npy"
    os.replace(emb_tmp, EMB_PATH)

    catalog_tmp = CATALOG_PATH + ".tmp"
    catalog.to_parquet(catalog_tmp, index=False)
    os.replace(catalog_tmp, CATALOG_PATH)

    index = build_index(normalized)
    index_tmp = INDEX_PATH + ".tmp"
    save_index(index, index_tmp)
    os.replace(index_tmp, INDEX_PATH)

    return normalized, index


def ingest_folder(audio_dir: str = AUDIO_DIR, preview_urls: dict = None, itunes_track_ids: dict = None) -> None:
    """Recursively scan `audio_dir`, embed any tracks not already in the catalog, and
    rebuild catalog.parquet / embeddings.npy / index.faiss. Idempotent: re-running with
    no new files leaves the catalog unchanged.

    `preview_urls`, if given, maps source_path -> iTunes previewUrl for any of the new
    files (used by `ingest_itunes`); files not in the map get `preview_url=None`.
    `itunes_track_ids` likewise maps source_path -> iTunes trackId (WEB_APP_SPEC.md §1);
    files not in the map get `itunes_track_id=None` (unknown, not "no id exists").

    Embeds in chunks of EMBED_BATCH_SIZE rather than stacking the whole batch into one
    array — at thousands of tracks, one giant np.stack() would be tens of GB of raw audio
    and risks OOM. Each chunk is persisted immediately, so a crash mid-run (OOM, ^C,
    network drop) only loses the current chunk, not everything ingested so far.
    """
    preview_urls = preview_urls or {}
    itunes_track_ids = itunes_track_ids or {}
    catalog = _load_catalog()
    embeddings = _load_embeddings()
    existing_paths = set(catalog["source_path"])

    candidates = [p for p in _find_audio_files(audio_dir) if p not in existing_paths]

    added = 0
    buffer_rows = []
    buffer_samples = []

    def flush_buffer():
        nonlocal catalog, embeddings, added
        if not buffer_samples:
            return
        chunk_embeddings = embed_batch(buffer_samples)
        embeddings = np.concatenate([embeddings, chunk_embeddings], axis=0)
        chunk_df = pd.DataFrame(buffer_rows)
        chunk_df["track_id"] = 0  # placeholder, reassigned below from row index
        catalog = pd.concat([catalog, chunk_df[list(CATALOG_DTYPES)]], ignore_index=True)
        catalog["track_id"] = catalog.index  # track_id = stable row index (SPEC.md §5)
        catalog["itunes_track_id"] = catalog["itunes_track_id"].astype("Int64")
        added += len(buffer_samples)
        _persist(catalog, embeddings)
        print(f"  checkpoint: {len(catalog)} total rows persisted ({added} new so far)")
        buffer_rows.clear()
        buffer_samples.clear()

    for path in tqdm(candidates, desc="Embedding audio"):
        try:
            samples = load_clip(path)
        except RuntimeError as e:
            # Decode failure: log and skip, don't crash the batch (SPEC.md §6).
            print(f"[skip] {e}")
            continue
        title, artist = _derive_title_artist(path)
        buffer_rows.append(
            {
                "title": title,
                "artist": artist,
                "source_path": path,
                "preview_url": preview_urls.get(path),
                "itunes_track_id": itunes_track_ids.get(path),
            }
        )
        buffer_samples.append(samples)
        if len(buffer_samples) >= EMBED_BATCH_SIZE:
            flush_buffer()

    flush_buffer()  # final partial chunk, if any

    # Always end with a fresh, consistent persist — covers the "nothing new" case too
    # (re-running with no new files still leaves catalog/embeddings/index in sync).
    normalized, index = _persist(catalog, embeddings)

    print(
        f"catalog: {len(catalog)} rows | embeddings: {normalized.shape} | "
        f"index: {index.ntotal} vectors | added {added} new track(s)"
    )


def ingest_itunes(csv_path: str) -> None:
    """Read a CSV of artist,title rows, fetch each track's ~30s iTunes preview into
    AUDIO_DIR, then embed via the same pipeline as ingest_folder. Skips rows whose
    preview file already exists (idempotent-ish, SPEC.md §6) and rows with no preview
    available, logging both rather than crashing the batch.

    This loop is rate-limited by ITUNES_SLEEP and can run for hours on a large CSV, so it
    calls ingest_folder every ITUNES_CHECKPOINT_EVERY downloads (not just once at the
    end) — if the run is interrupted, everything fetched up to the last checkpoint is
    already embedded and safely in the catalog, and re-running the same CSV resumes
    cleanly (already-downloaded previews are skipped via the on-disk existence check).
    """
    rows = pd.read_csv(csv_path)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    preview_urls = {}
    itunes_track_ids = {}
    total_downloaded = 0
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
        track_id = results[0].get("trackId")
        try:
            audio_resp = requests.get(preview_url, timeout=30)
            audio_resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[skip] {artist} - {title}: preview download failed ({e})")
            continue

        with open(out_path, "wb") as f:
            f.write(audio_resp.content)
        preview_urls[out_path] = preview_url
        itunes_track_ids[out_path] = track_id
        total_downloaded += 1
        print(f"[ok] {artist} - {title} -> {out_path}")

        if len(preview_urls) >= ITUNES_CHECKPOINT_EVERY:
            print(f"--- checkpoint: embedding {len(preview_urls)} downloaded track(s) so far ---")
            ingest_folder(AUDIO_DIR, preview_urls=preview_urls, itunes_track_ids=itunes_track_ids)
            preview_urls = {}
            itunes_track_ids = {}

        time.sleep(ITUNES_SLEEP)  # iTunes rate-limits around 20/min (SPEC.md §6)

    if preview_urls:
        ingest_folder(AUDIO_DIR, preview_urls=preview_urls, itunes_track_ids=itunes_track_ids)
    elif total_downloaded == 0:
        print("No new previews downloaded.")


def append_track(
    embedding: np.ndarray,
    *,
    title: str,
    artist: str,
    source_path: str = None,
    preview_url: str = None,
    itunes_track_id: int = None,
) -> int:
    """Append a single already-embedded track to catalog/embeddings.npy/index.faiss and
    persist, returning its new track_id. Used by the web app's cold path
    (WEB_APP_SPEC.md §2) so the append and the row-alignment invariant are enforced here,
    not duplicated in server.py.
    """
    catalog = _load_catalog()
    embeddings = _load_embeddings()

    row = {
        "title": title,
        "artist": artist,
        "source_path": source_path,
        "preview_url": preview_url,
        "itunes_track_id": itunes_track_id,
    }
    chunk_df = pd.DataFrame([row])
    chunk_df["track_id"] = 0  # placeholder, reassigned below from row index
    catalog = pd.concat([catalog, chunk_df[list(CATALOG_DTYPES)]], ignore_index=True)
    catalog["track_id"] = catalog.index
    catalog["itunes_track_id"] = catalog["itunes_track_id"].astype("Int64")

    embeddings = np.concatenate(
        [embeddings, np.asarray(embedding, dtype=np.float32).reshape(1, -1)], axis=0
    )

    _persist(catalog, embeddings)
    return int(catalog["track_id"].iloc[-1])
