from pathlib import Path


# ======== paths ========
DATA_DIR = Path("../data")
RAW_PARALLEL_CSV = DATA_DIR / "raw/parallel.csv"
RAW_LEXICON_CSV = DATA_DIR / "raw/lexicon.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PARALLEL_CLEANED_CSV = PROCESSED_DIR / "parallel_cleaned.csv"
LEXICON_CLEANED_CSV = PROCESSED_DIR / "lexicon_cleaned.csv"

CHECKPOINT_DIR = Path("../checkpoints")

# ======== model ========
MODEL_NAME = "t5-base"
MAX_SEQ_LEN = 128

# ======== hyperparameters ========
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ======== data ========
VALIDATION_SPLIT = 0.2
EVALUATION_SPLIT = 0.1
SHUFFLE = True
NUM_WORKERS = 4
PIN_MEMORY = True

# ======== pointer ========
USE_POINTER = True      # True for LexiconPointerNMT and False for BaseNMT
POINTER_THRESHOLD = 0.5