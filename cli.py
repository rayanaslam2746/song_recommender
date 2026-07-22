"""CLI entry point. See SPEC.md §7 for the target command surface.

`ingest` is wired up; `ingest-itunes` and `recommend` are not implemented yet.
"""

import argparse

from config import AUDIO_DIR
from src.ingest import ingest_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibe-based song recommender")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--audio-dir", default=AUDIO_DIR)

    subparsers.add_parser("ingest-itunes")
    subparsers.add_parser("recommend")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_folder(args.audio_dir)
        return
    if args.command is None:
        parser.print_help()
        return
    raise NotImplementedError(f"'{args.command}' is not implemented yet")


if __name__ == "__main__":
    main()
