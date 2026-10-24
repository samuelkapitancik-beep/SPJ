"""Metriky: ofic. skript organizátorov (vendorovaný) + bootstrap CI + McNemar.

Oficiálne funkcie sa importujú z `src/vendor/` (commit repozitára 6f71ea8, nemenené).
Rýchla vektorizovaná verzia (`score_arrays`) sa používa pre bootstrap a je testom
overená voči oficiálnemu skriptu.
"""
from __future__ import annotations
import importlib.util, math
from pathlib import Path
import numpy as np
from scipy.stats import binom
from .utils import ROOT


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

official_13 = _load("official_13", "src/vendor/official_eval_task1_3/evaluation_script.py")
official_24 = _load("official_24", "src/vendor/official_eval_task2_4/evaluation_script.py")


def official_scores(gold: list[dict], pred: list[dict]) -> dict:
    """Oficiálne ACC / TCE / combined (ST1). Vstup: gold s 'validity','plausibility'; pred s 'validity'."""
    gt_map = {x["id"]: x for x in gold}
    acc, _, _ = official_13.calculate_accuracy(gold, pred, "validity", "validity")
    cells = {}
    for key, (v, p) in {"acc_plausible_valid": (True, True), "acc_implausible_valid": (True, False),
                        "acc_plausible_invalid": (False, True), "acc_implausible_invalid": (False, False)}.items():
        cells[key], _, _ = official_13.calculate_subgroup_accuracy(gt_map, pred, v, p)
    bias = official_13.calculate_content_effect_bias(cells)
    tce = bias["tot_content_effect"]
    return {"accuracy": acc, "tce": tce, "tce_intra": bias["content_effect_intra_validity_label"],
            "tce_inter": bias["content_effect_inter_validity_label"],
            "combined_score": official_13.calculate_smooth_combined_metric(acc, tce), **cells}


def official_scores_st2(gold: list[dict], pred: list[dict]) -> dict:
    """Oficiálne ST2: Avg(ACC, F1_premises)/(1+ln(1+TCE)). pred má 'validity' a 'relevant_premises'."""
    gt_map = {x["id"]: x for x in gold}
    base = official_scores(gold, pred)
    f1 = official_24.calculate_f1_premises(gt_map, pred)
    avg = (base["accuracy"] + f1) / 2
    return {**base, "f1_premises": f1, "avg": avg,
            "combined_score": official_24.calculate_smooth_combined_metric(avg, base["tce"])}


# ---------- rýchla verzia pre bootstrap ----------
def score_arrays(y_true: np.ndarray, y_pred: np.ndarray, plaus: np.ndarray) -> dict:
    """Rovnaká metrika ako oficiálna (v %), nad numpy poľami bool."""
    acc = 100 * np.mean(y_true == y_pred)
    def cell(v, p):
        m = (y_true == v) & (plaus == p)
        return 100 * np.mean(y_pred[m] == y_true[m]) if m.any() else 0.0
    pv, iv, pi, ii = cell(True, True), cell(True, False), cell(False, True), cell(False, False)
    intra = (abs(pv - iv) + abs(pi - ii)) / 2
    inter = (abs(pv - pi) + abs(iv - ii)) / 2
    tce = (intra + inter) / 2
    return {"accuracy": acc, "tce": tce, "combined_score": acc / (1 + math.log(1 + tce))}


def to_arrays(gold: list[dict], pred: list[dict]):
    pm = {p["id"]: p["validity"] for p in pred}
    return (np.array([x["validity"] for x in gold]), np.array([pm[x["id"]] for x in gold]),
            np.array([x["plausibility"] for x in gold]))


def bootstrap_ci(gold: list[dict], pred: list[dict], n: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentilový bootstrap (resampling príkladov) pre accuracy, TCE a combined_score."""
    yt, yp, pl = to_arrays(gold, pred)
    rng = np.random.default_rng(seed)
    out = {"accuracy": [], "tce": [], "combined_score": []}
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        s = score_arrays(yt[i], yp[i], pl[i])
        for k in out: out[k].append(s[k])
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in out.items()}


def mcnemar_exact(gold: list[dict], pred_a: list[dict], pred_b: list[dict]) -> dict:
    """Presný (binomický) obojstranný McNemarov test na zhodných príkladoch."""
    yt, ya, _ = to_arrays(gold, pred_a)
    _, yb, _ = to_arrays(gold, pred_b)
    b = int(np.sum((ya == yt) & (yb != yt)))   # A správne, B nesprávne
    c = int(np.sum((ya != yt) & (yb == yt)))   # A nesprávne, B správne
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def confusion(gold: list[dict], pred: list[dict]) -> dict:
    yt, yp, _ = to_arrays(gold, pred)
    tp = int(np.sum(yt & yp)); tn = int(np.sum(~yt & ~yp)); fp = int(np.sum(~yt & yp)); fn = int(np.sum(yt & ~yp))
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": prec, "recall": rec, "f1_valid": f1}


def tce_noise_floor(accuracy: float, n_per_cell: int = 48, sims: int = 20000, seed: int = 0) -> dict:
    """Očakávané TCE modelu, ktorého chyby sú NEZÁVISLÉ od validity aj plausibility (skutočný content effect = 0),
    pri danej celkovej presnosti. Ukazuje, koľko TCE vzniká iba vzorkovacím šumom pri malých bunkách."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(n_per_cell, accuracy / 100, size=(sims, 4)) / n_per_cell * 100
    pv, iv, pi, ii = a.T
    tce = (((abs(pv - iv) + abs(pi - ii)) / 2) + ((abs(pv - pi) + abs(iv - ii)) / 2)) / 2
    cs = accuracy / (1 + np.log1p(tce))
    return {"accuracy": accuracy, "tce_mean": float(tce.mean()), "tce_p05": float(np.percentile(tce, 5)),
            "tce_p95": float(np.percentile(tce, 95)), "cs_mean": float(cs.mean())}
