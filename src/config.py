from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SLIDES = ROOT / "slides"
RESULTS = ROOT / "results"

for p in (ART, SLIDES, RESULTS):
    p.mkdir(parents=True, exist_ok=True)