"""Načítanie, validácia a rozdelenie dát SemEval-2026 Task 11."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import ROOT

REQUIRED = {"id", "syllogism", "validity", "plausibility"}
CELLS = [(True, True), (True, False), (False, True), (False, False)]  # (validity, plausibility)


def load_json(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def validate(data: list[dict], name: str = "dataset", need_premises: bool = False) -> None:
    """Zastaví beh pri porušení schémy (chýbajúce polia, nebooleovské štítky, duplicitné id)."""
    required = REQUIRED | ({"relevant_premises"} if need_premises else set())
    ids = set()
    for i, x in enumerate(data):
        missing = required - x.keys()
        if missing:
            raise ValueError(f"{name}[{i}] chýbajú polia {missing}")
        if not isinstance(x["validity"], bool) or not isinstance(x["plausibility"], bool):
            raise ValueError(f"{name}[{i}] validity/plausibility musia byť bool")
        if not x["syllogism"].strip():
            raise ValueError(f"{name}[{i}] prázdny text")
        if x["id"] in ids:
            raise ValueError(f"{name}: duplicitné id {x['id']}")
        ids.add(x["id"])


def normalize_text(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def duplicates(data: list[dict]) -> int:
    norm = [normalize_text(x["syllogism"]) for x in data]
    return len(norm) - len(set(norm))


def overlap(a: list[dict], b: list[dict]) -> set[str]:
    """Normalizované texty, ktoré sú v oboch množinách (kontrola dátového úniku)."""
    return {normalize_text(x["syllogism"]) for x in a} & {normalize_text(x["syllogism"]) for x in b}


def cell_of(x: dict) -> str:
    return f"{'valid' if x['validity'] else 'invalid'}_{'plaus' if x['plausibility'] else 'implaus'}"


def cell_counts(data: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in data:
        out[cell_of(x)] = out.get(cell_of(x), 0) + 1
    return dict(sorted(out.items()))


def stratified_split(data: list[dict], dev_fraction: float, seed: int):
    """Train/dev rozdelenie stratifikované podľa 4 buniek (validity x plausibility)."""
    strata = [cell_of(x) for x in data]
    idx = np.arange(len(data))
    tr, dv = train_test_split(idx, test_size=dev_fraction, random_state=seed, stratify=strata)
    return [data[i] for i in sorted(tr)], [data[i] for i in sorted(dv)]


def prepare_splits(cfg: dict):
    """Vráti (train, dev, test_st1, test_st2, report). Test sa nikdy nepoužíva na voľbu hyperparametrov."""
    train_all = load_json(cfg["paths"]["train"])
    test1 = load_json(cfg["paths"]["test_st1"])
    test2 = load_json(cfg["paths"]["test_st2"])
    validate(train_all, "train"); validate(test1, "test_st1"); validate(test2, "test_st2", need_premises=True)

    report = {
        "n_train_raw": len(train_all), "n_test_st1": len(test1), "n_test_st2": len(test2),
        "duplicates_train": duplicates(train_all), "duplicates_test_st1": duplicates(test1),
        "overlap_train_test_st1": sorted(overlap(train_all, test1)),
        "overlap_train_test_st2": sorted(overlap(train_all, test2)),
    }
    if cfg["split"]["remove_test_overlap"]:
        bad = overlap(train_all, test1) | overlap(train_all, test2)
        train_all = [x for x in train_all if normalize_text(x["syllogism"]) not in bad]
    report["n_train_after_filter"] = len(train_all)
    train, dev = stratified_split(train_all, cfg["split"]["dev_fraction"], cfg["seed"])
    report.update(n_train=len(train), n_dev=len(dev),
                  cells_train=cell_counts(train), cells_dev=cell_counts(dev),
                  cells_test_st1=cell_counts(test1))
    assert not (set(x["id"] for x in train) & set(x["id"] for x in dev)), "únik train->dev"
    return train, dev, test1, test2, report
