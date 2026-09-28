import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Base paths for datasets and experiments. Environment variables allow cluster
# jobs to override the local layout without editing tracked source files.
LIBERO_DATASET_PATH = os.environ.get(
    "LIBERO_DATASET_PATH", str(PROJECT_ROOT / "datasets" / "libero")
)
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", str(PROJECT_ROOT / "outputs"))
