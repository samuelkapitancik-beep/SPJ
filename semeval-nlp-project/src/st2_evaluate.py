"""ST2 metriky: rozpad podľa vetiev a buniek, analýza remíz, rýchla vektorizovaná metrika a bootstrap.

Pôvodný src/evaluate.py sa NEMENÍ - oficiálne skóre ST2 (`official_scores_st2`) a ST1 rýchla metrika
(`score_arrays`) sa odtiaľ iba importujú. Rýchla ST2 metrika je testom overená voči oficiálnemu skriptu.
"""
from __future__ import annotations
import math
import numpy as np
from .evaluate import official_scores_st2, score_arrays, official_24   # noqa: F401 (official_24 pre run_st2/testy)


def premise_pr(gold_prem: list[int], pred_prem: list[int]) -> tuple[float, float]:
    """Precision a recall premís pre jeden príklad presne ako v oficiálnom `calculate_f1_premises`."""
    t, p = set(gold_prem), set(pred_prem)
    tp, fp, fn = len(t & p), len(p - t), len(t - p)
    return (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)


def st2_arrays(gold: list[dict], pred: list[dict]):
    """Polia po príkladoch: validity gold/pred, plausibility, precision, recall, gold-platný (vstupuje do F1)."""
    pm = {p["id"]: p for p in pred}
    yt = np.array([x["validity"] for x in gold]); yp = np.array([pm[x["id"]]["validity"] for x in gold])
    pl = np.array([x["plausibility"] for x in gold])
    pr = np.array([premise_pr(x["relevant_premises"], pm[x["id"]]["relevant_premises"]) for x in gold])
    gv = np.array([len(x["relevant_premises"]) > 0 for x in gold])
    return yt, yp, pl, pr[:, 0], pr[:, 1], gv


def score_arrays_st2(yt, yp, pl, prec, rec, gv) -> dict:
    """Rovnaká metrika ako oficiálny skript ST2 (v %), nad numpy poľami. Overené testom voči oficiálnemu skriptu."""
    s = score_arrays(yt, yp, pl)
    if gv.any():
        mp, mr = prec[gv].mean(), rec[gv].mean()
        f1 = 100 * 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    else:
        f1 = 0.0
    avg = (s["accuracy"] + f1) / 2
    return {"accuracy": s["accuracy"], "f1_premises": f1, "tce": s["tce"], "avg": avg,
            "combined_score": avg / (1 + math.log(1 + s["tce"]))}


def bootstrap_ci_st2(gold: list[dict], pred: list[dict], n: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentilový bootstrap (resampling príkladov) pre ACC, F1 premís, TCE a combined_score ST2."""
    arr = st2_arrays(gold, pred)
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "f1_premises", "tce", "combined_score")
    out = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(arr[0]), len(arr[0]))
        s = score_arrays_st2(*(a[i] for a in arr))
        for k in keys: out[k].append(s[k])
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in out.items()}


def st2_breakdown(gold: list[dict], pred: list[dict], decisions=None) -> dict:
    """Diagnostika ST2: presnosť po bunkách, chyby typu valid->invalid (stoja ACC aj F1), a ak sú dané
    symbolické rozhodnutia, podiel a presnosť jednotlivých vetiev (valid-pair / invalid-sure / undecided)
    a analýza remíz (viac platných dvojíc)."""
    pm = {p["id"]: p for p in pred}
    missing = sum(pm[x["id"]]["validity"] is None for x in gold)
    if missing:                       # ako ST1 (neural_common.evaluate_predictions): abstencia nie je INVALID
        return {"n": len(gold), "unparsable_answers": missing, "answer_coverage": (len(gold) - missing) / len(gold),
                "accuracy_with_abstentions_as_errors":
                    100 * sum(pm[x["id"]]["validity"] == x["validity"] for x in gold) / len(gold),
                "official_withheld": True}
    s = official_scores_st2(gold, pred)
    out = {k: s[k] for k in ("accuracy", "f1_premises", "tce", "combined_score", "acc_plausible_valid",
                             "acc_implausible_valid", "acc_plausible_invalid", "acc_implausible_invalid")}
    out["n"] = len(gold)
    out["valid_predicted_invalid"] = sum(x["validity"] and not pm[x["id"]]["validity"] for x in gold)
    out["invalid_predicted_valid"] = sum(not x["validity"] and pm[x["id"]]["validity"] for x in gold)
    out["valid_correct_wrong_pair"] = sum(x["validity"] and pm[x["id"]]["validity"] and
                                          set(pm[x["id"]]["relevant_premises"]) != set(x["relevant_premises"]) for x in gold)
    if decisions is None:
        return out
    branches = {}
    for x, d in zip(gold, decisions):
        b = branches.setdefault(d.status, {"n": 0, "correct": 0, "gold_valid": 0, "reasons": {}})
        b["n"] += 1; b["correct"] += pm[x["id"]]["validity"] == x["validity"]; b["gold_valid"] += x["validity"]
        if d.reason:
            b["reasons"][d.reason] = b["reasons"].get(d.reason, 0) + 1
    for b in branches.values():
        b["accuracy"] = 100 * b["correct"] / b["n"]
    out["branches"] = dict(sorted(branches.items()))
    ties = [(x, d) for x, d in zip(gold, decisions) if d.tie]
    out["ties"] = {
        "n": len(ties),
        "gold_valid": sum(x["validity"] for x, _ in ties),
        # remízy, kde je správna dvojica medzi platnými - iba na nich má zmysel porovnávať pravidlá pre remízy
        "gold_pair_among_valid": sum(x["validity"] and tuple(sorted(x["relevant_premises"])) in d.valid_pairs for x, d in ties),
        "chosen_pair_correct": sum(x["validity"] and set(d.premises) == set(x["relevant_premises"]) for x, d in ties),
    }
    return out
