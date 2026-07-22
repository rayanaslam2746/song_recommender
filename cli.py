"""CLI entry point. See SPEC.md §7 for the target command surface.

Recommendations are ranked purely by embedding similarity — no artist de-dup/cap.
"""

import argparse
import sys

import numpy as np
import pandas as pd

# ingest_itunes prints raw artist/title strings, which can be any script (Hangul,
# Devanagari, Tamil, ...) now that build_lists.py pulls non-English tracks. Windows'
# default console codepage (cp1252) can't encode most of that and crashes on print();
# force UTF-8 stdout/stderr with a safe fallback instead of erroring mid-run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import AUDIO_DIR, CATALOG_PATH, EMB_PATH
from src.ingest import ingest_folder, ingest_itunes
from src.recommend import recommend, recommend_from_file, recommend_from_search


def _print_results(results: list) -> None:
    if not results:
        print("No results.")
        return
    print(f"{'rank':>4}  {'score':>7}  {'title':<40}  {'artist':<30}")
    for rank, r in enumerate(results, start=1):
        print(f"{rank:>4}  {r['score']:>7.4f}  {str(r['title'])[:40]:<40}  {str(r['artist'])[:30]:<30}")


def _recommend_by_track_id(track_id: int, k: int) -> list:
    catalog = pd.read_parquet(CATALOG_PATH)
    matches = catalog.index[catalog["track_id"] == track_id]
    if len(matches) == 0:
        raise SystemExit(f"track_id {track_id} not found in catalog")
    row_pos = int(matches[0])

    embeddings = np.load(EMB_PATH)
    query_vector = embeddings[row_pos]

    return recommend(query_vector, k=k, exclude_track_id=track_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibe-based song recommender")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--audio-dir", default=AUDIO_DIR)

    ingest_itunes_parser = subparsers.add_parser("ingest-itunes")
    ingest_itunes_parser.add_argument("--csv", required=True)

    recommend_parser = subparsers.add_parser("recommend")
    query_group = recommend_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--track-id", type=int)
    query_group.add_argument("--file", type=str)
    query_group.add_argument("--search", type=str)
    recommend_parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_folder(args.audio_dir)
        return

    if args.command == "ingest-itunes":
        ingest_itunes(args.csv)
        return

    if args.command == "recommend":
        if args.track_id is not None:
            results = _recommend_by_track_id(args.track_id, args.k)
        elif args.file is not None:
            results = recommend_from_file(args.file, k=args.k)
        else:
            results = recommend_from_search(args.search, k=args.k)
        _print_results(results)
        return

    if args.command is None:
        parser.print_help()
        return
    raise NotImplementedError(f"'{args.command}' is not implemented yet")


if __name__ == "__main__":
    main()
