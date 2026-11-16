''' 
Project-wide constants and configs
'''
from pathlib import Path

# Project paths
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJ_ROOT / "output"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

CONFIG_DIR = PROJ_ROOT / "configs"

# Reproducibility
DEFAULT_RANDOM_SEED = 910