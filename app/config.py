# config.py (multi-task)
from pathlib import Path
import os

# === Paths ===
CSV_PATH   = Path(os.getenv("CSV_PATH", "../app_data/DATA.csv"))
DATA_DIR   = Path(os.getenv("DATA_DIR",  "../app_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR","../models"))
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# === Targets ===
TARGET_COLS = [
    'Did hard CHD occured?(1=Present,0=Absent)',  # chd
    'CVD(1=Present,0=Absent)',                    # cvd
    'CKD(1=Present,0=Absent)',                    # ckd
    'DM(1=Present,0=Absent)',                     # dm
    'HMOD(1=Present,0=Absent)',                   # hmod
]

HEADS = ["chd", "cvd", "ckd", "dm", "hmod"]

# Columns to drop if any for later
DROP_COLS = []

# Exclude all target labels from inputs
EXCLUDE_AS_FEATURES = TARGET_COLS[:]

# FL settings
NUM_CLIENTS    = 4
ROUNDS         = 5
LOCAL_EPOCHS   = 3
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3

# Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_INFER_REQUESTS = os.getenv("TOPIC_INFER_REQUESTS", "ai.infer.requests")
TOPIC_RISK_CLASSIFICATIONS = os.getenv("TOPIC_RISK_CLASSIFICATIONS", "ai.risk.classifications")

# Risk-tiering (used primarily for CHD)
PERCENTILE_WINDOW = 5000
