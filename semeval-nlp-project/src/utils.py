"""Pomocné funkcie: seed, konfigurácia, ukladanie výsledkov, záznam prostredia."""
from __future__ import annotations
import json, platform, random, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path = "configs/experiment.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def environment_info() -> dict:
    import numpy, pandas, scipy, sklearn
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
