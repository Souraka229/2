"""Configuration DataTour 2026 — détection fraude mobile money."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Fichiers détectés automatiquement (noms avec ou sans suffixe)
TRAIN_CANDIDATES = ["train.csv", "train (4).csv"]
TEST_CANDIDATES = ["test.csv", "test (6).csv"]
SAMPLE_CANDIDATES = ["sample_submission.csv", "sample_submission (2).csv", "sample_submission (1).csv"]


def _resolve(candidates: list[str]) -> Path:
    for name in candidates:
        path = ROOT / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Aucun fichier trouvé parmi : {candidates}")


TRAIN_PATH = _resolve(TRAIN_CANDIDATES)
TEST_PATH = _resolve(TEST_CANDIDATES)
SAMPLE_PATH = _resolve(SAMPLE_CANDIDATES)

OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
SUBMISSION_PATH = ROOT / "submission.csv"

ID_COL = "id"
TARGET_COL = "fraud_flag"
SUBMIT_COL = "target"

# Validation temporelle : test commence à period 106
VAL_PERIOD_CUTOFF = 96
ROLLING_CUTOFFS = [75, 81, 87, 93, 99]
RANDOM_STATE = 42

# Target encoding — lissage bayésien
TE_SMOOTHING = 50

# Post-traitement op_03 : validé LB (+0.003 vs v3 sans filtre)
USE_OPERATION_PRIOR = True

# Plancher d'itérations full-train (évite sous-apprentissage si early stopping trop agressif)
MIN_FULL_ITERS = {"lgbm": 800, "catboost": 600, "xgboost": 600, "op03": 800}
