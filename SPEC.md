# SPEC.md — Vibe-Based Song Recommender (MVP)

## 0. What we're building

A song recommender that matches tracks purely by **how they sound** — genre, timbre, groove, energy, "vibe" — and **not** by artist, language, title, popularity, or any other metadata.

The acid test: a melodic Tamil R&B track (e.g. "Unakkul Naanae") should surface Bryson Tiller–style tracks as neighbors, because they occupy the same sonic space. Two songs by the same artist that sound nothing alike should **not** be treated as similar.

This is a **content-based audio-similarity** system, not collaborative filtering. There is no user history and no metadata in the similarity math. Metadata (artist/title) is stored for display and de-duplication only — never for scoring.

## 1. Core design principles (do not violate)

1. **Similarity is computed only from audio embeddings.** Artist, language, title, genre tags, year — none of these ever enter the distance calculation.
2. **The embedding model must be language- and artist-agnostic by construction.** We use a pretrained audio encoder that only ever sees the waveform.
3. **De-duplicate by artist at recommendation time**, not at scoring time. We find the nearest-sounding tracks first, then optionally cap how many come from any single artist so results are diverse and sound-driven (directly addresses the "don't just give me more of the same artist" requirement).

## 2. Tech stack

- **Language:** Python 3.10+
- **Embeddings:** [`laion_clap`](https://github.com/LAION-AI/CLAP) (LAION-CLAP) — contrastive audio encoder, 512-dim output, strong at musical "vibe."
- **Vector search:** `faiss-cpu` (exact flat index for MVP).
- **Audio I/O:** `librosa` + `soundfile` (needs system `ffmpeg` for mp3/m4a decoding).
- **Catalog store:** `pandas` + Parquet.
- **Preview fetching (optional):** `requests` against the free iTunes Search API.
- **CLI:** `argparse` (or `typer` if preferred).
- **Utilities:** `numpy`, `tqdm`.

CPU-only is fine for the MVP. GPU just makes embedding faster.

## 3. Model & metric decisions (locked)

**Embedding model.** Use `laion_clap` with the **music-pretrained checkpoint** `music_audioset_epoch_15_esc_90.pt`, loaded as:

```python
import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
model.load_ckpt('music_audioset_epoch_15_esc_90.pt')  # download once from LAION's release
```

Fallback if that checkpoint gives load errors: `model = laion_clap.CLAP_Module(enable_fusion=False)` then `model.load_ckpt()` (no args → default general audio checkpoint). Note the difference in a comment; the music checkpoint gives noticeably better musical matches.

Embeddings are **512-dimensional float32**.

**Distance metric.** Cosine similarity. Implement it in FAISS as inner-product on L2-normalized vectors:
- L2-normalize every embedding before indexing (`faiss.normalize_L2`).
- Use `faiss.IndexFlatIP(512)`.
- Higher inner-product score = more similar.

Do **not** use raw Euclidean on un-normalized vectors — magnitude carries loudness/length artifacts we don't want.

**Clip strategy.** For each track, extract a single **mono, 48 kHz, 30-second clip taken from the middle of the file** (30s previews are used whole). Feed the raw samples to CLAP via `get_audio_embedding_from_data(x=samples, use_tensor=False)` so we control exactly what gets embedded. (Multi-window averaging is a v1.1 improvement — see §11.)

## 4. Repository structure

```
song-vibe-recommender/
├── README.md
├── requirements.txt
├── config.py                 # all constants & paths
├── cli.py                    # entry point
├── data/
│   ├── audio/                # raw audio files / downloaded previews
│   ├── embeddings.npy        # (N, 512) float32, row-aligned with catalog
│   ├── catalog.parquet       # metadata, one row per track, row i ↔ vector i
│   └── index.faiss           # persisted FAISS index
├── src/
│   ├── __init__.py
│   ├── audio.py              # load → mono 48k → center 30s clip
│   ├── embed.py              # CLAP wrapper: samples → 512-d vector
│   ├── ingest.py            # build catalog from a folder and/or iTunes CSV
│   ├── index_store.py        # build / save / load FAISS index
│   └── recommend.py          # nearest-neighbor query + artist de-dup
└── tests/
    └── test_smoke.py
```

## 5. Data model

`catalog.parquet` — one row per track, **row order must match** `embeddings.npy` rows and FAISS insertion order:

| column        | type   | notes                                             |
|---------------|--------|---------------------------------------------------|
| `track_id`    | int    | stable id = row index                             |
| `title`       | str    | display only                                      |
| `artist`      | str    | display + de-dup only, never scored               |
| `source_path` | str    | path to the audio file used                       |
| `preview_url` | str    | nullable; set if fetched from iTunes              |

`embeddings.npy` — `(N, 512)` float32, **L2-normalized** before it's saved (so the index and any cached copy agree).

## 6. Component specs

### `src/audio.py`
- `load_clip(path: str) -> np.ndarray`
  - `librosa.load(path, sr=48000, mono=True)`.
  - If longer than 30s, slice the centered 30s window; if shorter, use the whole thing (optionally zero-pad to a small minimum).
  - Return float32 samples. Raise a clear error on decode failure (log path, skip in batch jobs — don't crash the whole ingest).

### `src/embed.py`
- Lazy-load the CLAP model once (module-level singleton).
- `embed_samples(samples: np.ndarray) -> np.ndarray` → returns `(512,)` float32.
- `embed_batch(list_of_samples) -> np.ndarray` → `(B, 512)`; batch for speed.
- Do **not** normalize here; normalization happens in `index_store` so there's one source of truth.

### `src/ingest.py`
Two entry paths, both producing/appending to `catalog.parquet` + `embeddings.npy`:
- `ingest_folder(audio_dir)` — recursively scan for audio files, derive `title`/`artist` from filename or tags (best-effort; display only), embed each, append rows.
- `ingest_itunes(csv_path)` — CSV with columns `artist,title`. For each row:
  - GET `https://itunes.apple.com/search?term=<artist title>&entity=song&limit=1`
  - Take `results[0].previewUrl` (a ~30s clip), download to `data/audio/`.
  - Sleep ~3s between requests (iTunes rate-limits around 20/min). Skip + log rows with no preview.
- Both paths must be **idempotent-ish**: skip tracks whose `source_path` is already in the catalog so re-runs don't duplicate.

### `src/index_store.py`
- `build_index(embeddings) -> faiss.Index`: copy embeddings, `faiss.normalize_L2(...)`, add to `faiss.IndexFlatIP(512)`.
- `save_index(index, path)` / `load_index(path)` via `faiss.write_index` / `read_index`.
- Rebuild the index from `embeddings.npy` whenever new tracks are ingested (flat index is cheap to rebuild for MVP scale).

### `src/recommend.py`
- `recommend(query_vector, k=10, max_per_artist=1, exclude_track_id=None, exclude_artist=None) -> list[dict]`
  - Normalize the query vector, `index.search(query, k * 8)` (over-fetch so de-dup still leaves enough).
  - Drop the query track itself (`exclude_track_id`) and any `exclude_artist`.
  - Walk results in score order, enforcing `max_per_artist` (default 1 → at most one track per artist).
  - Return top `k` as dicts: `{track_id, title, artist, score, preview_url}`.
- `recommend_from_file(path, ...)` and `recommend_from_search(query_str, ...)`: embed a brand-new track on the fly (not necessarily in the catalog), then run `recommend`.

## 7. CLI (`cli.py`)

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

Output: a clean ranked table — rank, score, title, artist. Print the score so we can eyeball match quality during tuning.

## 8. `config.py` constants

```python
SAMPLE_RATE   = 48000
CLIP_SECONDS  = 30
EMBED_DIM     = 512
CLAP_CKPT     = "music_audioset_epoch_15_esc_90.pt"
MAX_PER_ARTIST_DEFAULT = 1
DATA_DIR      = "data"
AUDIO_DIR     = "data/audio"
EMB_PATH      = "data/embeddings.npy"
CATALOG_PATH  = "data/catalog.parquet"
INDEX_PATH    = "data/index.faiss"
ITUNES_SLEEP  = 3.0   # seconds between preview fetches
```

## 9. Setup / README steps

1. `python -m venv .venv && source .venv/bin/activate`
2. Install system `ffmpeg` (`brew install ffmpeg` / `apt install ffmpeg`).
3. `pip install -r requirements.txt`
4. Download the CLAP music checkpoint `music_audioset_epoch_15_esc_90.pt` from LAION's CLAP release and place it in the repo root (README should link the release page and note it's ~2 GB).
5. Drop some audio into `data/audio/` (or prepare `songs.csv`), then run an `ingest` command.
6. Run a `recommend` command and eyeball the neighbors.

`requirements.txt` should pin `laion_clap`, `torch`, `faiss-cpu`, `librosa`, `soundfile`, `numpy`, `pandas`, `pyarrow`, `requests`, `tqdm`. Let CLAP pull its own compatible `torch`/`transformers`; if there's a resolver conflict, pin `torch` to a CPU wheel and document it.

## 10. Known pitfalls (call these out in code comments)

- **CLAP dependency friction.** `laion_clap` is picky about `torch`/`transformers` versions. Build inside a fresh venv; if `load_ckpt` throws, try the default checkpoint fallback (§3) before anything else.
- **ffmpeg required** for mp3/m4a decode. librosa will fail silently-ish without it.
- **Sample rate must be 48 kHz** going into CLAP. Resample in `load_clip`, don't assume the source rate.
- **Row alignment is sacred.** `catalog.parquet` row i, `embeddings.npy` row i, and FAISS vector i must always refer to the same track. Any ingest that appends must append to all three consistently (or rebuild the index from the npy after appending catalog+embeddings).
- **Normalize exactly once**, in `index_store`, for both the indexed vectors and every query vector. Mismatched normalization silently ruins scores.
- **iTunes coverage gaps** on long-tail/regional tracks — some queries return no `previewUrl`. Skip and log; don't crash the batch.

## 11. Definition of done (MVP acceptance criteria)

- [ ] `ingest --audio-dir` embeds a folder and produces `catalog.parquet`, `embeddings.npy`, `index.faiss`.
- [ ] `ingest-itunes --csv` downloads previews and adds them to the catalog.
- [ ] `recommend --track-id` returns K neighbors ranked by cosine score, respecting `--max-per-artist`.
- [ ] `recommend --file` and `--search` work on tracks not already in the catalog.
- [ ] Similarity uses **only** embeddings — grep the code and confirm no artist/title/genre value ever touches the search or scoring.
- [ ] Smoke test in `tests/test_smoke.py`: embed 2 clearly-similar clips and 1 clearly-different clip, assert the similar pair scores higher than either does against the outlier.
- [ ] Manual sanity check: on a ~100–200 track catalog spanning several genres, top neighbors "feel right" and cross artist/language lines when the sound matches.

## 12. Stretch goals (v1.1 — out of MVP scope)

- **Multi-window embedding:** average 3 windows (early/middle/late) per track, then normalize, for a more robust whole-song fingerprint.
- **Swap-in OpenL3** as an alternate encoder behind the same `embed_samples` interface, to A/B match quality.
- **Approximate index** (`IndexHNSWFlat` or IVF+PQ) once the catalog exceeds ~1M vectors.
- **Minimal UI:** a Streamlit page — search a song, hear the 30s preview, see neighbors with play buttons.
- **Tunable blend:** optional light re-rank that nudges by tempo/key extracted with librosa, kept strictly separate from the core embedding search and off by default.
