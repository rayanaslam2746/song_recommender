"""One-off migration: add a nullable `itunes_track_id` column to an existing
catalog.parquet without disturbing row order (WEB_APP_SPEC.md §1).

Existing rows get itunes_track_id = <NA> (unknown, not "no id exists") — ingest_itunes
and the web app's cold path populate it for every track added from here on.
"""

import pandas as pd

from config import CATALOG_PATH


def main() -> None:
    catalog = pd.read_parquet(CATALOG_PATH)
    before_rows = len(catalog)

    if "itunes_track_id" in catalog.columns:
        print(f"itunes_track_id already present ({before_rows} rows) — no changes made.")
        return

    catalog["itunes_track_id"] = pd.array([pd.NA] * before_rows, dtype="Int64")
    catalog.to_parquet(CATALOG_PATH, index=False)

    after = pd.read_parquet(CATALOG_PATH)
    assert len(after) == before_rows, "row count changed during migration"
    assert list(after["track_id"]) == list(range(before_rows)), "row order/track_id changed during migration"
    print(f"Added itunes_track_id column ({before_rows} rows, row order preserved).")


if __name__ == "__main__":
    main()
