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
