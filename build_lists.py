"""Build a language-diverse, deduplicated artist,title,language CSV ready to feed into
`cli.py ingest-itunes --csv`.

English/Western tracks come from Last.fm's tag.getTopTracks (BUILD_LISTS_SPEC.md).
Non-English tracks come from Apple's iTunes RSS regional charts (LANGUAGE_DIVERSITY_SPEC.md)
— Last.fm's catalog skews Western, so its regional-language tags are thin and lead to a
second coverage loss downstream in ingest_itunes; iTunes charts are the same catalog
ingest_itunes searches, so previews match almost perfectly.

This script only produces a CSV of names — no audio, embedding, or indexing happens here;
that's all downstream in ingest_itunes.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

# Chart/error content can include non-Latin scripts; force UTF-8 stdout/stderr so a
# printed name doesn't crash on Windows' default cp1252 console (see cli.py for the same
# fix, hit for real during testing when ingest_itunes printed a Hangul title).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    ENGLISH_SHARE,
    GENRE_IDS,
    GENRES,
    LASTFM_BASE,
    LASTFM_CACHE_DIR,
    LASTFM_PAGE_SIZE,
    LASTFM_SLEEP,
    NON_ENGLISH_WEIGHTS,
    REGION_COUNTRIES,
    SONGS_CSV,
    TOTAL_TARGET,
)
from src.itunes_charts import fetch_chart


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
    """Dedup on a normalized artist|title key, keeping the first occurrence. Works on
    (artist, title, ...) rows of any length, so it dedups across English (3-field) and
    language-tagged (4-field) rows alike — a track counted once doesn't also count toward
    a second language's quota (LANGUAGE_DIVERSITY_SPEC.md §5).
    """
    seen = set()
    deduped = []
    dropped = 0
    for row in all_tracks:
        artist, title = row[0], row[1]
        key = f"{artist.strip().lower()}|{title.strip().lower()}"
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def _collect_language(language: str, target: int, countries: list, genre_id: str = None) -> list:
    """Collect up to `target` (artist, title, source_tag) tuples for one non-English
    language across all its listed countries, stopping once the target is hit or every
    country is exhausted. Charts are shallow (~100 max per call, see ITUNES_RSS_MAX_LIMIT)
    — a target far beyond that is expected to fall short; that's logged, not retried
    (refetching the same chart returns the same snapshot, not new songs).
    """
    collected = []
    for country in countries:
        if len(collected) >= target:
            break
        pairs = fetch_chart(country, limit=target - len(collected), genre_id=genre_id)
        tag = f"{country}-genre-{language}" if genre_id else f"{country}-chart"
        print(f"  [{language}] {country}: {len(pairs)} tracks ({tag})")
        collected.extend((artist, title, tag) for artist, title in pairs)

    if len(collected) < target:
        print(f"[shortfall] {language}: got {len(collected)} / {target} (chart(s) exhausted, not retrying)")

    return collected[:target]


def _write_csv(rows: list, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "title", "source_genre", "language"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a language-diverse, deduplicated artist,title,language CSV"
    )
    parser.add_argument("--out", default=SONGS_CSV)
    parser.add_argument("--total", type=int, default=TOTAL_TARGET)
    parser.add_argument("--english-share", type=float, default=ENGLISH_SHARE)
    parser.add_argument("--genres", nargs="+", default=GENRES)
    parser.add_argument(
        "--per-genre", type=int, default=None,
        help="override the computed per-genre Last.fm target (default: derived from --total/--english-share)",
    )
    parser.add_argument(
        "--only-language", choices=sorted(NON_ENGLISH_WEIGHTS), default=None,
        help="skip Last.fm/English entirely and fetch just this one non-English language (--total is the direct target for it)",
    )
    args = parser.parse_args()

    all_tracks = []  # (artist, title, source_tag, language)

    if args.only_language:
        lang = args.only_language
        countries = REGION_COUNTRIES[lang]
        genre_id = GENRE_IDS.get(lang)
        print(f"=== --only-language {lang}: target {args.total}, countries {countries} ===")
        triples = _collect_language(lang, args.total, countries, genre_id=genre_id)
        all_tracks.extend((artist, title, tag, lang) for artist, title, tag in triples)
    else:
        english_target = round(args.total * args.english_share)
        non_english_total = args.total - english_target
        language_targets = {
            lang: round(non_english_total * weight) for lang, weight in NON_ENGLISH_WEIGHTS.items()
        }
        per_genre = args.per_genre or max(1, round(english_target / len(args.genres)))

        print("=== computed targets ===")
        print(f"total: {args.total} | english_share: {args.english_share}")
        print(f"english: {english_target} target ({per_genre}/genre across {len(args.genres)} Last.fm genres)")
        for lang, t in language_targets.items():
            print(f"  {lang}: {t}")
        print()

        api_key = os.environ.get("LASTFM_API_KEY")
        if not api_key:
            print(
                "LASTFM_API_KEY is not set. Get a free key at "
                "https://www.last.fm/api/account/create and set it as an environment "
                "variable before running this script."
            )
            sys.exit(1)

        for genre in args.genres:
            try:
                pairs = _collect_genre(genre, api_key, per_genre)
            except LastfmAuthError as e:
                print(f"Last.fm rejected the API key: {e}")
                sys.exit(1)
            print(f"[{genre}] collected {len(pairs)} / {per_genre}")
            all_tracks.extend((artist, title, genre, "english") for artist, title in pairs)

        print()
        for lang, target in language_targets.items():
            if target <= 0:
                continue
            countries = REGION_COUNTRIES.get(lang, [])
            genre_id = GENRE_IDS.get(lang)
            genre_note = f" genre={genre_id}" if genre_id else ""
            print(f"=== {lang}: target {target}, countries {countries}{genre_note} ===")
            triples = _collect_language(lang, target, countries, genre_id=genre_id)
            all_tracks.extend((artist, title, tag, lang) for artist, title, tag in triples)

    total_before = len(all_tracks)
    deduped, dropped = _dedup(all_tracks)
    total_after = len(deduped)

    _write_csv(deduped, args.out)

    print()
    print(f"Total before dedup: {total_before}")
    print(f"Duplicates dropped: {dropped}")
    print(f"Total after dedup: {total_after}")

    print()
    print("Actual language distribution after dedup:")
    lang_counts = {}
    for row in deduped:
        lang = row[3]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    for lang, count in sorted(lang_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / total_after if total_after else 0.0
        print(f"  {lang}: {count} ({pct:.1f}%)")

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
