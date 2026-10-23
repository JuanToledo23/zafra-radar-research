"""Project paths. All data directories are created on import."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
NISAR_DIR = DATA_DIR / "nisar"
S1_DIR = DATA_DIR / "sentinel1"
FIRMS_DIR = DATA_DIR / "firms"
FIELDS_DIR = DATA_DIR / "fields"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_DIR, NISAR_DIR, S1_DIR, FIRMS_DIR, FIELDS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
