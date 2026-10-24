"""Export predikcií v oficiálnom formáte a overenie oficiálnym skriptom."""
from __future__ import annotations
import json, tempfile, os
from pathlib import Path
from .evaluate import official_13


def export_predictions(preds: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)


def run_official_file_scoring(gold: list[dict], preds: list[dict]) -> dict:
    """Spustí PRIAMO oficiálny `run_full_scoring` nad súbormi (end-to-end kontrola formátu)."""
    with tempfile.TemporaryDirectory() as d:
        g, p, o = (os.path.join(d, n) for n in ("gold.json", "pred.json", "out.json"))
        json.dump(gold, open(g, "w")); json.dump(preds, open(p, "w"))
        official_13.run_full_scoring(g, p, o)
        return json.load(open(o))
