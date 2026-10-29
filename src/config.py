"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MITDB = DATA / "mitdb" / "mit-bih-arrhythmia-database-1.0.0"
NSTDB = DATA / "nstdb"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "report"

for _p in (PROCESSED, RESULTS, MODELS, FIGURES, REPORT):
    _p.mkdir(parents=True, exist_ok=True)

FS = 360
WIN_BEFORE = 96          # 0.267 s before the R peak
WIN_AFTER = 160          # 0.444 s after the R peak
WIN = WIN_BEFORE + WIN_AFTER  # 256 samples = 0.711 s

CLASS_NAMES = ["N", "S", "V", "F"]
CLASS_DESC = {
    "N": "Normal / LBBB / RBBB / escape / junctional",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion",
}
NUM_CLASSES = len(CLASS_NAMES)

# AAMI-style grouping.  Q (unknown/paced/unclassifiable) is excluded from the
# supervised task; see the report for the rationale.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}
LABEL_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

# de Chazal inter-patient split (22 records for training / 22 for testing).
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

SEED = 42
