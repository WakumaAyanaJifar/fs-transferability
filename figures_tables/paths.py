"""Repository-relative paths shared by the figure and table scripts.

Fresh outputs from 04_model.py / 05_revision.py (outputs/...) are used when present;
otherwise the archived aggregated results in results/ are used.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pick(sub):
    fresh = ROOT / "outputs" / sub
    return fresh if fresh.exists() else ROOT / "results" / sub


MODEL = _pick("model")          # 04_model.py outputs
REV = _pick("revision")         # 05_revision.py outputs
RAW = ROOT / "data" / "raw"
FIG = ROOT / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
