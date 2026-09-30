import os
from pathlib import Path

# Base paths for datasets and experiments
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_DATASET_PATH = os.environ.get("LIBERO_DATASET_PATH", str(PROJECT_ROOT / "data/libero"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(PROJECT_ROOT / "output"))
