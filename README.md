# song-vibe-recommender

A song recommender that matches tracks purely by how they sound — genre, timbre,
groove, energy, "vibe" — never by artist, language, title, or popularity. Content-based
audio similarity via [LAION-CLAP](https://github.com/LAION-AI/CLAP) embeddings + FAISS.
See [SPEC.md](SPEC.md) for the full design.

## Setup

1. Create and activate a virtualenv:
   ```
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```
2. Install system `ffmpeg` (needed for mp3/m4a decoding):
   ```
   brew install ffmpeg      # macOS
   apt install ffmpeg       # Debian/Ubuntu
   ```
   On Windows, install via [ffmpeg.org](https://ffmpeg.org/download.html) or
   `winget install ffmpeg` and make sure `ffmpeg` is on `PATH`.
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the CLAP music checkpoint `music_audioset_epoch_15_esc_90.pt` from the
   [LAION-CLAP releases page](https://github.com/LAION-AI/CLAP/releases) (~2 GB) and
   place it in the repo root.
5. Drop some audio into `data/audio/` (or prepare a `songs.csv` with `artist,title`
   columns), then run an `ingest` command.
6. Run a `recommend` command and eyeball the neighbors.

## CLI

```
# Build the catalog + index from a folder of audio files
python cli.py ingest --audio-dir data/audio

# Build from an iTunes CSV (artist,title per row) — downloads 30s previews
python cli.py ingest-itunes --csv songs.csv

# Recommend by an existing catalog track
python cli.py recommend --track-id 42 --k 10

# Recommend from a brand-new local file not in the catalog
python cli.py recommend --file ~/Downloads/some_song.mp3 --k 10

# Recommend from an iTunes search (fetch preview, embed, search)
python cli.py recommend --search "Anirudh Unakkul Naanae" --k 10
```

## Web UI

A small local web app (see [WEB_APP_SPEC.md](WEB_APP_SPEC.md)) so you can search for a
song, pick a match, and hear sound-alike recommendations in the browser — no CLI, no
audio files to handle yourself.

If you have a catalog from before this feature existed, backfill the new
`itunes_track_id` column once (safe to re-run; no-ops if already present):
```
python migrate_add_itunes_track_id.py
```

Then start the server and open the page:
```
python -m uvicorn server:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000
```

The first startup takes ~30s while the CLAP checkpoint loads (once, at process start —
every query after that is ~3-4s). The server must stay running while the page is in use.
Searching a song already in the catalog returns recommendations near-instantly; a new
song gets downloaded, embedded, and added to the catalog on the spot, so searching it
again later is instant too.

## Status

`src/audio.py`, `src/embed.py`, `src/index_store.py`, `src/recommend.py`, both paths of
`src/ingest.py` (`ingest_folder` and `ingest_itunes`), and the FastAPI web UI
(`server.py` + `static/index.html`) are functional.

Recommendations are ranked purely by embedding similarity, with no artist de-dup/cap
(SPEC.md §1.3).
