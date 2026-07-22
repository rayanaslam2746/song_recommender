"""Fetch top-song charts from Apple's iTunes RSS feeds. No API key needed.

Two feed hosts, both verified live against real responses (LANGUAGE_DIVERSITY_SPEC.md §2):

- Classic: https://itunes.apple.com/{country}/rss/topsongs/limit={n}/json
  Supports an optional genre filter (.../genre={genre_id}/json) — the only way to get
  Tamil/Telugu specifically, since India's unfiltered chart is a mixed, Hindi-leaning bag
  rather than a clean single-language feed. Entries live at feed.entry[], which is a
  *single dict* (not a list) when there's exactly one result — must parse defensively.
  Caps out around 83-100 real entries even when a much higher `limit` is requested;
  requesting too high (~300) returns HTTP 400.
- Newer (fallback): https://rss.marketingtools.apple.com/api/v2/{country}/music/most-played/{n}/songs.json
  Entries at feed.results[], always a list (no single/list ambiguity). Reliable up to
  limit=100; higher fails with a 500. Does not support genre filtering, so it's a
  fallback for the unfiltered-country-chart case only, not for Tamil/Telugu.
"""

import time

import requests

from config import ITUNES_CHART_SLEEP, ITUNES_RSS_MAX_LIMIT

CLASSIC_BASE = "https://itunes.apple.com/{country}/rss/topsongs/limit={n}/json"
CLASSIC_GENRE_BASE = "https://itunes.apple.com/{country}/rss/topsongs/limit={n}/genre={genre_id}/json"
FALLBACK_BASE = "https://rss.marketingtools.apple.com/api/v2/{country}/music/most-played/{n}/songs.json"


def _parse_classic(data: dict) -> list:
    entries = data.get("feed", {}).get("entry", [])
    if isinstance(entries, dict):
        # Single-result quirk: entry is a bare object, not a 1-item list.
        entries = [entries]

    pairs = []
    for e in entries:
        name = e.get("im:name", {}).get("label")
        artist = e.get("im:artist", {}).get("label")
        if name and artist:
            pairs.append((artist, name))
    return pairs


def _parse_fallback(data: dict) -> list:
    results = data.get("feed", {}).get("results", [])
    pairs = []
    for r in results:
        name = r.get("name")
        artist = r.get("artistName")
        if name and artist:
            pairs.append((artist, name))
    return pairs


def fetch_chart(country: str, limit: int = None, genre_id: str = None) -> list:
    """Fetch up to `limit` (artist, title) pairs from one country's top-songs chart.

    If `genre_id` is given, filters to that Apple Music genre via the classic feed only
    — the fallback feed has no genre filter, so genre-filtered requests never fall back.

    For unfiltered requests, the classic feed isn't uniformly reliable across countries —
    e.g. it returned only 4 (wrong-looking) entries for `kr` in testing, while the newer
    fallback feed gave a full 100 for the same country. So rather than only falling back
    on total failure, we top up from the fallback feed whenever classic under-delivers,
    deduping against what classic already gave.
    """
    limit = min(limit or ITUNES_RSS_MAX_LIMIT, ITUNES_RSS_MAX_LIMIT)

    classic_pairs = []
    if genre_id is not None:
        url = CLASSIC_GENRE_BASE.format(country=country, n=limit, genre_id=genre_id)
    else:
        url = CLASSIC_BASE.format(country=country, n=limit)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        classic_pairs = _parse_classic(resp.json())
        time.sleep(ITUNES_CHART_SLEEP)
    except (requests.RequestException, ValueError) as e:
        print(f"[warn] classic chart fetch failed for {country} (genre={genre_id}): {e}")

    if genre_id is not None:
        return classic_pairs[:limit]  # nothing sensible to fall back to

    if len(classic_pairs) >= limit:
        return classic_pairs[:limit]

    try:
        url = FALLBACK_BASE.format(country=country, n=limit)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        fallback_pairs = _parse_fallback(resp.json())
        time.sleep(ITUNES_CHART_SLEEP)
    except (requests.RequestException, ValueError) as e:
        print(f"[error] fallback chart fetch also failed for {country}: {e}")
        fallback_pairs = []

    seen = {(a.strip().lower(), t.strip().lower()) for a, t in classic_pairs}
    combined = list(classic_pairs)
    for artist, title in fallback_pairs:
        key = (artist.strip().lower(), title.strip().lower())
        if key not in seen:
            seen.add(key)
            combined.append((artist, title))

    if fallback_pairs:
        print(f"    ({country}: classic gave {len(classic_pairs)}, topped up to {min(len(combined), limit)} via fallback feed)")

    return combined[:limit]
