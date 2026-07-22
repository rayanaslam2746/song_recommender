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
python cli.py recommend --track-id 42 --k 10 --max-per-artist 1

# Recommend from a brand-new local file not in the catalog
python cli.py recommend --file ~/Downloads/some_song.mp3 --k 10

# Recommend from an iTunes search (fetch preview, embed, search)
python cli.py recommend --search "Anirudh Unakkul Naanae" --k 10
```

## Status

Early scaffold — `src/audio.py` and `src/embed.py` are functional (load a clip, produce
a 512-dim CLAP embedding). `ingest`, `index_store`, and `recommend` are stubs, not yet
implemented.
