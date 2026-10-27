"""Project paths. All data directories are created on import."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
NISAR_DIR = DATA_DIR / "nisar"
S1_DIR = DATA_DIR / "sentinel1"
FIRMS_DIR = DATA_DIR / "firms"
FIELDS_DIR = DATA_DIR / "fields"
LABELS_DIR = DATA_DIR / "labels"
FIRES_DIR = DATA_DIR / "fires"  # derived per-field fire tables (raw FIRMS CSVs: FIRMS_DIR)
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"  # site configurations (zafra.site)
OUTPUTS_DIR = ROOT / "outputs"

for _d in (
    DATA_DIR,
    NISAR_DIR,
    S1_DIR,
    FIRMS_DIR,
    FIELDS_DIR,
    LABELS_DIR,
    FIRES_DIR,
    FEATURES_DIR,
    OUTPUTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
