"""All constants & paths for the song recommender."""

SAMPLE_RATE = 48000
CLIP_SECONDS = 30
EMBED_DIM = 512
CLAP_CKPT = "music_audioset_epoch_15_esc_90.pt"
DATA_DIR = "data"
AUDIO_DIR = "data/audio"
EMB_PATH = "data/embeddings.npy"
CATALOG_PATH = "data/catalog.parquet"
INDEX_PATH = "data/index.faiss"
ITUNES_SLEEP = 3.0  # seconds between preview fetches

# A single np.stack() over the whole new-track batch is fine at small scale, but at
# thousands of tracks it becomes tens of GB of raw float32 audio in one array (SAMPLE_RATE
# * CLIP_SECONDS * 4 bytes per clip) and risks OOM. Embed in bounded chunks instead, and
# checkpoint catalog/embeddings/index after each one so a crash mid-run only loses the
# current chunk, not the whole ingest.
EMBED_BATCH_SIZE = 32
# For ingest_itunes' long fetch loop (rate-limited by ITUNES_SLEEP, can run for hours):
# re-embed and checkpoint every this-many successful downloads, rather than only once at
# the very end, so a crash partway through a long overnight run doesn't lose everything
# fetched so far.
ITUNES_CHECKPOINT_EVERY = 100

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
LASTFM_SLEEP = 0.25  # seconds between Last.fm requests
LASTFM_PAGE_SIZE = 250  # tracks per page request
LASTFM_CACHE_DIR = "data/.lastfm_cache"
SONGS_CSV = "data/songs.csv"

# Hand-picked, tempo- and texture-diverse genres (not Last.fm's raw "top tags", which are
# polluted with non-genre tags like "seen live" and "favorites"). BUILD_LISTS_SPEC.md §3.
GENRES = [
    "rock", "pop", "hip-hop", "rnb", "electronic", "house", "techno",
    "jazz", "classical", "metal", "country", "folk", "indie", "soul",
    "reggae", "blues", "punk", "ambient", "latin", "k-pop",
]
TRACKS_PER_GENRE = 500

# --- Language diversity (LANGUAGE_DIVERSITY_SPEC.md) ---
TOTAL_TARGET = 10000  # rough overall size
ENGLISH_SHARE = 0.70  # 70% English/Western, via the existing Last.fm genre path

# Non-English allocation (sums to ~1.0), applied to the 30% remainder, via iTunes RSS
# regional charts (§1: Last.fm's catalog skews Western, so these come from Apple's own
# charts instead — better coverage, and near-perfect ingest_itunes preview matches later
# since it's the same catalog).
NON_ENGLISH_WEIGHTS = {
    "hindi": 0.24,
    "tamil": 0.17,
    "telugu": 0.13,
    "korean": 0.17,
    "japanese": 0.17,
    "chinese": 0.12,
}

ITUNES_CHART_SLEEP = 0.5  # polite pause between iTunes chart fetches
# Verified live: both RSS feeds return real data up to this depth (classic feed 400s
# above ~200, actual entry count plateaus around 83-100; fallback feed 500s above 100).
# Requesting more doesn't yield more real tracks — charts are just this shallow.
ITUNES_RSS_MAX_LIMIT = 100

REGION_COUNTRIES = {  # language -> country code(s), tried in order until target is hit
    "hindi": ["in"], "tamil": ["in"], "telugu": ["in"],
    "korean": ["kr"], "japanese": ["jp"], "chinese": ["tw", "cn"],
}

# Apple Music genre IDs for Tamil/Telugu (needed because India's unfiltered chart is
# Hindi/Punjabi-leaning, per REGION_COUNTRIES). Verified live via the `category` field on
# real `in`-country chart responses (https://itunes.apple.com/in/rss/topsongs/limit=50/json
# and the genre-filtered variant), not guessed from memory (BUILD_LISTS_SPEC.md §4 warns
# explicitly against hardcoding these). 1263=Bollywood and 1262=Indian were also found
# during the same lookup, in case Hindi ever needs its own filter later.
GENRE_IDS = {
    "tamil": "1264",
    "telugu": "1265",
}
