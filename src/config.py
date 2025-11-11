from pathlib import Path


# =========================
# Paths
# =========================
DATA_DIR = Path("../data")
RAW_PARALLEL_CSV = DATA_DIR / "raw/parallel.csv"
RAW_LEXICON_CSV = DATA_DIR / "raw/lexicon.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PARALLEL_CLEANED_CSV = PROCESSED_DIR / "parallel_cleaned.csv"
LEXICON_CLEANED_CSV = PROCESSED_DIR / "lexicon_cleaned.csv"

SAVE_DIR = Path("../checkpoints")

# =============
# Model
# =============
MODEL_NAME = "t5-base"
MAX_SEQ_LEN = 128

# ===================
# Training Parameters
# ===================
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5

# =============
# Data
# =============
VALIDATION_SPLIT = 0.2
EVALUATION_SPLIT = 0.1
SHUFFLE = True
NUM_WORKERS = 4
PIN_MEMORY = True

# =============
# Pointer
# =============
USE_POINTER = True
POINTER_THRESHOLD = 0.5