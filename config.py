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
