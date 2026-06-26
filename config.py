"""Пути и константы проекта. Значения читаются из config/config.yaml."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Загружает конфиг проекта из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["data_dir"]
FIGURES_DIR = PROJECT_ROOT / CONFIG["paths"]["figures_dir"]
RAW_DATA_FILE = DATA_DIR / CONFIG["paths"]["raw_data_file"]
CLEANED_DATA_FILE = DATA_DIR / CONFIG["paths"]["cleaned_data_file"]
NORMALIZED_DATA_FILE = DATA_DIR / CONFIG["paths"]["normalized_data_file"]
FEATURES_LIST_FILE = DATA_DIR / CONFIG["paths"]["features_list_file"]
SCALER_FILE = DATA_DIR / CONFIG["paths"]["scaler_file"]
CLUSTERED_DATA_FILE = DATA_DIR / CONFIG["paths"]["clustered_data_file"]
BEST_MODEL_FILE = DATA_DIR / CONFIG["paths"]["best_model_file"]

RANDOM_STATE = CONFIG["model"]["random_state"]
N_CLUSTERS = CONFIG["clustering"]["n_clusters"]
