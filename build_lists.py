"""Pull the top N tracks for each of ~20 genres from the Last.fm API and write a single
deduplicated artist,title CSV ready to feed into `cli.py ingest-itunes --csv`.

See BUILD_LISTS_SPEC.md. This script only produces a CSV of names — no audio, embedding,
or indexing happens here; that's all downstream in ingest_itunes.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

from config import (
    GENRES,
    LASTFM_BASE,
    LASTFM_CACHE_DIR,
    LASTFM_PAGE_SIZE,
    LASTFM_SLEEP,
    SONGS_CSV,
    TRACKS_PER_GENRE,
)


class LastfmAuthError(Exception):
    """Last.fm error code 10: invalid API key. Fatal — retrying won't help."""


def _cache_path(genre: str, page: int) -> str:
    safe_genre = re.sub(r"[^a-z0-9\-]", "_", genre.lower())
    return os.path.join(LASTFM_CACHE_DIR, f"{safe_genre}_p{page}.json")


def _fetch_page(genre: str, page: int, api_key: str) -> dict:
    """Fetch one page of tag.getTopTracks, using the on-disk cache if present.

    Retries with exponential backoff on network errors and rate-limiting (error 29).
    Raises LastfmAuthError on an invalid key (code 10). Any other Last.fm error is
    logged and treated as "nothing more for this tag" (returns {}).
    """
    cache_path = _cache_path(genre, page)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "method": "tag.getTopTracks",
        "tag": genre,
        "api_key": api_key,
        "format": "json",
        "limit": LASTFM_PAGE_SIZE,
        "page": page,
    }

    max_attempts = 5
    backoff = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(LASTFM_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            if attempt == max_attempts:
                print(f"[error] {genre} page {page}: network error after {max_attempts} attempts ({e})")
                return {}
            print(f"[retry] {genre} page {page}: network error ({e}), waiting {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue

        if "error" in data:
            code = data.get("error")
            message = data.get("message", "")
            if code == 10:
                raise LastfmAuthError(message or "invalid API key")
            if code == 29:
                if attempt == max_attempts:
                    print(f"[error] {genre} page {page}: still rate-limited after {max_attempts} attempts, giving up")
                    return {}
                print(f"[retry] {genre} page {page}: rate-limited (29), waiting {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"[skip] {genre} page {page}: Last.fm error {code} ({message})")
            return {}

        os.makedirs(LASTFM_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        time.sleep(LASTFM_SLEEP)  # only throttle real requests, not cache hits
        return data

    return {}


def _parse_tracks(data: dict) -> list:
    """Extract (artist, title) pairs from a tag.getTopTracks response, defensively."""
    track_list = data.get("tracks", {}).get("track", [])
    if isinstance(track_list, dict):
        # Last.fm returns a bare dict instead of a 1-item list when there's only one track.
        track_list = [track_list]

    pairs = []
    for t in track_list:
        name = t.get("name")
        artist = t.get("artist")
        artist_name = artist.get("name") if isinstance(artist, dict) else artist
        if name and artist_name:
            pairs.append((artist_name, name))
    return pairs


def _collect_genre(genre: str, api_key: str, target_count: int) -> list:
    collected = []
    page = 1
    while len(collected) < target_count:
        data = _fetch_page(genre, page, api_key)
        if not data:
            break

        pairs = _parse_tracks(data)
        if not pairs:
            break
        collected.extend(pairs)

        attr = data.get("tracks", {}).get("@attr", {})
        try:
            total_pages = int(attr.get("totalPages", page))
        except (TypeError, ValueError):
            total_pages = page

        if page >= total_pages:
            break
        page += 1

    return collected[:target_count]


def _dedup(all_tracks: list) -> tuple:
    """Dedup on a normalized artist|title key, keeping the first occurrence."""
    seen = set()
    deduped = []
    dropped = 0
    for artist, title, genre in all_tracks:
        key = f"{artist.strip().lower()}|{title.strip().lower()}"
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append((artist, title, genre))
    return deduped, dropped


def _write_csv(rows: list, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "title", "source_genre"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deduplicated artist,title CSV from Last.fm genre charts")
    parser.add_argument("--out", default=SONGS_CSV)
    parser.add_argument("--per-genre", type=int, default=TRACKS_PER_GENRE)
    parser.add_argument("--genres", nargs="+", default=GENRES)
    args = parser.parse_args()

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print(
            "LASTFM_API_KEY is not set. Get a free key at "
            "https://www.last.fm/api/account/create and set it as an environment "
            "variable before running this script."
        )
        sys.exit(1)

    all_tracks = []  # (artist, title, genre)
    per_genre_counts = {}

    for genre in args.genres:
        try:
            pairs = _collect_genre(genre, api_key, args.per_genre)
        except LastfmAuthError as e:
            print(f"Last.fm rejected the API key: {e}")
            sys.exit(1)

        per_genre_counts[genre] = len(pairs)
        print(f"[{genre}] collected {len(pairs)} / {args.per_genre}")
        all_tracks.extend((artist, title, genre) for artist, title in pairs)

    total_before = len(all_tracks)
    deduped, dropped = _dedup(all_tracks)
    total_after = len(deduped)

    _write_csv(deduped, args.out)

    print()
    print("Per-genre collected counts:")
    for genre, count in per_genre_counts.items():
        print(f"  {genre}: {count}")
    print(f"Total before dedup: {total_before}")
    print(f"Duplicates dropped: {dropped}")
    print(f"Total after dedup: {total_after}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
