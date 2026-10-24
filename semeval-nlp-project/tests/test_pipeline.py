import math, random
import numpy as np
import pytest
from src.data_loader import load_json, validate, stratified_split, cell_counts, prepare_splits, normalize_text, overlap
from src.evaluate import official_scores, score_arrays, to_arrays, mcnemar_exact, official_scores_st2
from src.predict import run_official_file_scoring
from src.preprocessing import split_sentences, n_premises
from src.baselines import plausibility_heuristic, majority_baseline, random_premises
from src.utils import load_config

CFG = load_config()


def _fake(n=192, seed=0):
    r = random.Random(seed)
    gold = [{"id": str(i), "validity": i % 2 == 0, "plausibility": (i // 2) % 2 == 0, "syllogism": "x"} for i in range(n)]
    pred = [{"id": g["id"], "validity": (g["validity"] if r.random() < .8 else not g["validity"])} for g in gold]
    return gold, pred


def test_fast_metric_equals_official():
    for seed in range(5):
        gold, pred = _fake(seed=seed)
        off = official_scores(gold, pred)
        fast = score_arrays(*to_arrays(gold, pred))
        for k in ("accuracy", "tce", "combined_score"):
            assert fast[k] == pytest.approx(off[k])


def test_official_file_scoring_matches():
    gold, pred = _fake()
    f = run_official_file_scoring(gold, pred)
    off = official_scores(gold, pred)
    assert f["accuracy"] == pytest.approx(off["accuracy"], abs=1e-3)
    assert f["content_effect"] == pytest.approx(off["tce"], abs=1e-3)
    assert f["combined_score"] == pytest.approx(off["combined_score"], abs=1e-3)


def test_perfect_and_heuristic_reference_points():
    gold, _ = _fake()
    perfect = [{"id": g["id"], "validity": g["validity"]} for g in gold]
    s = official_scores(gold, perfect)
    assert (s["accuracy"], s["tce"], s["combined_score"]) == (100, 0, 100)
    h = official_scores(gold, plausibility_heuristic(gold))
    assert h["accuracy"] == pytest.approx(50) and h["combined_score"] < 10


def test_data_validation_rejects_bad_label():
    with pytest.raises(ValueError):
        validate([{"id": "1", "syllogism": "a", "validity": "yes", "plausibility": True}])
    with pytest.raises(ValueError):
        validate([{"id": "1", "syllogism": "a", "validity": True, "plausibility": True},
                  {"id": "1", "syllogism": "b", "validity": True, "plausibility": True}])


def test_split_is_stratified_disjoint_and_deterministic():
    tr = load_json(CFG["paths"]["train"])
    a1, b1 = stratified_split(tr, 0.2, 42); a2, b2 = stratified_split(tr, 0.2, 42)
    assert [x["id"] for x in b1] == [x["id"] for x in b2]
    assert not ({x["id"] for x in a1} & {x["id"] for x in b1})
    assert len(a1) + len(b1) == len(tr)
    c = cell_counts(b1)
    assert max(c.values()) - min(c.values()) <= 3  # dev je takmer vyvážený


def test_no_train_test_leakage_after_prepare():
    train, dev, t1, t2, rep = prepare_splits(CFG)
    assert not overlap(train + dev, t1) and not overlap(train + dev, t2)
    assert rep["n_train"] + rep["n_dev"] == rep["n_train_after_filter"]


def test_sentences_and_premises():
    t = "All a are b. Some c are a. Therefore, some c are b."
    assert len(split_sentences(t)) == 3 and n_premises(t) == 2


def test_random_premises_valid_indices_and_deterministic():
    t2 = load_json(CFG["paths"]["test_st2"])
    r1, r2 = random_premises(t2, 1), random_premises(t2, 1)
    assert r1 == r2
    assert all(len(set(r)) == 2 and max(r) < n_premises(x["syllogism"]) for r, x in zip(r1, t2))


def test_st2_gold_premise_indices_are_zero_based_premises():
    t2 = load_json(CFG["paths"]["test_st2"])
    for x in t2:
        if x["validity"]:
            assert len(x["relevant_premises"]) == 2
            assert max(x["relevant_premises"]) < n_premises(x["syllogism"])


def test_mcnemar_identical_models_p1():
    gold, pred = _fake()
    assert mcnemar_exact(gold, pred, pred)["p_value"] == 1.0


def test_noise_floor_monotone_and_positive():
    from src.evaluate import tce_noise_floor
    lo, hi = tce_noise_floor(60), tce_noise_floor(95)
    assert lo["tce_mean"] > 0 and lo["tce_mean"] > hi["tce_mean"]   # vyššia presnosť -> menší šum


def test_cv_oof_covers_every_example_once():
    from src.baselines import cv_oof_predictions, build_tfidf_lr
    tr = load_json(CFG["paths"]["train"])[:200]
    oof = cv_oof_predictions(tr, lambda: build_tfidf_lr("word", (1, 1), 1.0, 500, 0), 5, 0)
    assert [o["id"] for o in oof] == [x["id"] for x in tr] and all(isinstance(o["validity"], bool) for o in oof)


# ------------------------------------------------------------- symbolická časť (parser, abstrakcia, platnosť)
def test_24_valid_forms_with_existential_import():
    from src.symbolic import count_valid_forms
    assert count_valid_forms() == 24


def test_validity_known_schemes():
    from src.symbolic import is_valid_stmts
    assert is_valid_stmts([("A", "Z", "Y"), ("A", "X", "Z"), ("A", "X", "Y")])        # Barbara
    assert is_valid_stmts([("E", "Z", "Y"), ("A", "X", "Z"), ("E", "X", "Y")])        # Celarent
    assert not is_valid_stmts([("A", "Y", "Z"), ("A", "X", "Z"), ("A", "X", "Y")])    # nerozložený stredný termín


def test_tokenizer_and_lemma():
    from src.symbolic import tokenize, lemma
    assert tokenize("Not all cats can't fly.") == ["not", "all", "cats", "can't", "fly", "."]
    assert lemma("roses") == lemma("rose") and lemma("classes") == lemma("class") and lemma("buses") == lemma("bus")


def test_statement_types():
    from src.symbolic import statement_type
    cases = {"All cars are vehicles.": "A", "No animal is a car.": "E", "Some dogs are mammals.": "I",
             "Some dogs are not cats.": "O", "Not all cats are black.": "O", "There are no ants that do not have six legs.": "A",
             "It must be the case that some sadness is not a feeling.": "O", "It is certain that no fish are mammals.": "E",
             "Nothing that is a fiction book is a biography.": "E"}
    for s, t in cases.items():
        assert statement_type(s)[0] == t, s


def test_parse_abstract_canonical_and_determinism():
    from src.symbolic import parse_syllogism, abstract_text, canonical_text, is_valid_stmts
    t = "All cats are animals. Every kitten is a cat. Therefore, all kittens are animals."
    p = parse_syllogism(t)
    assert p.ok and is_valid_stmts(p.stmts)
    ab = abstract_text(t, p)
    assert all(w not in ab.lower() for w in ("cat", "animal", "kitten")) and set("XYZ") <= set(ab)
    assert canonical_text(p) == canonical_text(parse_syllogism(t))
    assert canonical_text(p).splitlines()[2] == "Conclusion: All X are Y."


def test_parser_abstains_instead_of_guessing():
    from src.symbolic import parse_syllogism
    assert not parse_syllogism("Dogs bark. Cats meow.").ok          # nie 3 vety
    assert not parse_syllogism("Birds sing. Fish swim. Trees grow.").ok


def test_plausibility_invariance_of_symbolic_decision():
    """Rovnaká forma s iným obsahom musí dať rovnakú predikciu (kontrafaktuálna invariantnosť)."""
    from src.symbolic import parse_syllogism, is_valid_stmts
    a = "All cats are animals. Every kitten is a cat. Therefore, all kittens are animals."
    b = "All rocks are living beings. Every pebble is a rock. Therefore, all pebbles are living beings."
    pa, pb = parse_syllogism(a), parse_syllogism(b)
    assert pa.stmts == pb.stmts and is_valid_stmts(pa.stmts) == is_valid_stmts(pb.stmts)


def test_extract_label():
    from src.neurosymbolic import extract_label
    assert extract_label("VALID") is True and extract_label(" invalid.") is False
    assert extract_label("It is valid, not invalid") is None and extract_label("maybe") is None


def test_prompt_contains_form_only_instruction_and_no_leak():
    from src.neurosymbolic import make_messages, build_item
    it = build_item({"id": "1", "syllogism": "All cats are animals. Every kitten is a cat. Therefore, all kittens are animals.",
                     "validity": True, "plausibility": True})
    msgs = make_messages(it, "abstract")
    assert "cat" not in msgs[1]["content"].lower().replace("categor", "") and "VALID or INVALID" in msgs[1]["content"]
    assert "plausib" not in msgs[1]["content"].lower()


def test_tokenizer_keeps_non_ascii_letters():
    from src.symbolic import tokenize
    assert tokenize("Všetky psy sú zvieratá.") == ["všetky", "psy", "sú", "zvieratá", "."]
    assert tokenize("Alle Hunde sind Säugetiere.")[-2] == "säugetiere"
