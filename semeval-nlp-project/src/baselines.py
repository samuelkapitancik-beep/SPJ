"""Baseline modely. Všetky vracajú zoznam predikcií v oficiálnom formáte [{'id', 'validity'}]."""
from __future__ import annotations
import itertools, random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .evaluate import official_scores
from .preprocessing import n_premises


def majority_baseline(train: list[dict], data: list[dict]) -> list[dict]:
    """B0a: vždy väčšinová trieda z trénovacích dát (pri zhode True)."""
    pos = sum(x["validity"] for x in train)
    label = pos >= len(train) - pos
    return [{"id": x["id"], "validity": label} for x in data]


def plausibility_heuristic(data: list[dict]) -> list[dict]:
    """B0b: DIAGNOSTICKÁ referencia "platné == vierohodné". Používa zlatý štítok plausibility,
    takže nejde o reálny model; ukazuje najhoršie možné správanie s maximálnym content effectom."""
    return [{"id": x["id"], "validity": x["plausibility"]} for x in data]


def random_premises(data: list[dict], seed: int) -> list[list[int]]:
    """Referencia pre výber premís (ST2): náhodná dvojica z n premís (0-indexované)."""
    rng = random.Random(seed)
    return [sorted(rng.sample(range(n_premises(x["syllogism"])), 2)) for x in data]


def build_tfidf_lr(analyzer: str, ngram_range, C: float, max_iter: int, seed: int) -> Pipeline:
    """B1: TF-IDF + logistická regresia. Vektorizér sa fituje iba na tréningových dátach (je v Pipeline)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range), lowercase=True, sublinear_tf=True)),
        ("lr", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)),
    ])


def predict_with(model: Pipeline, data: list[dict]) -> list[dict]:
    labels = model.predict([x["syllogism"] for x in data])
    return [{"id": x["id"], "validity": bool(y)} for x, y in zip(data, labels)]


def cv_oof_predictions(train: list[dict], build, folds: int, seed: int) -> list[dict]:
    """Pooled out-of-fold predikcie: každý trénovací príklad je predikovaný modelom, ktorý ho nevidel."""
    X = np.array([x["syllogism"] for x in train], dtype=object)
    y = np.array([x["validity"] for x in train])
    strata = [f"{x['validity']}_{x['plausibility']}" for x in train]
    oof = [None] * len(train)
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, strata):
        m = build(); m.fit(list(X[tr]), list(y[tr]))
        for i, lab in zip(va, m.predict(list(X[va]))):
            oof[i] = {"id": train[i]["id"], "validity": bool(lab)}
    return oof


def grid_search_cv(train: list[dict], grid: dict, max_iter: int, seed: int, folds: int = 5):
    """Voľba hyperparametrov podľa pooled-OOF combined_score na TRAIN (n~766, ~190 príkladov/bunka)."""
    results = []
    for an, ng, C in itertools.product(grid["analyzer"], grid["ngram_range"], grid["C"]):
        oof = cv_oof_predictions(train, lambda: build_tfidf_lr(an, ng, C, max_iter, seed), folds, seed)
        s = official_scores(train, oof)
        results.append({"analyzer": an, "ngram_range": list(ng), "C": C,
                        "cv_accuracy": s["accuracy"], "cv_tce": s["tce"], "cv_combined": s["combined_score"]})
    best = max(results, key=lambda r: (r["cv_combined"], r["cv_accuracy"]))
    return best, results
