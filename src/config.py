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
MODEL_NAME = "google/flan-t5-small" # or google/t5-efficient-mini flan-t5-small
DROPOUT_RATE = 0.2
MAX_SEQ_LEN = 87

# ======== hyperparameters ========
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
FREEZE_LAYERS = True

# ======== data ========
VALIDATION_SPLIT = 0.2
EVALUATION_SPLIT = 0.1
SHUFFLE = True
NUM_WORKERS = 4
PIN_MEMORY = True

# ======== pointer ========
USE_POINTER = True      # True for LexiconPointerNMT and False for BaseNMT
POINTER_THRESHOLD = 0.5

# ======== early stopping ========
EARLY_STOPPING = True
EARLY_STOPPING_MIN_DELTA = 0.001