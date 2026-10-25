"""Testy ST2: symetria parsera, generátor dev množín, symbolický rozhodovač S2 a zhoda metriky s oficiálnym skriptom."""
import json, os, random, tempfile
import numpy as np
import pytest
from src.data_loader import load_json
from src.preprocessing import split_sentences
from src.symbolic import parse_syllogism, is_valid_stmts
from src.st2_devset import make_st2_dev, make_st2_like, VARIANTS
from src.st2_symbolic import solve, predict, overlap_pair
from src.evaluate import official_scores_st2, official_24
from src.st2_evaluate import st2_arrays, score_arrays_st2, st2_breakdown

TRAIN = load_json("data/processed/train.json")
DEV = load_json("data/processed/dev.json")

VALID = ("All dogs are mammals. Some cars are red. All mammals are animals. No rivers are mountains. "
         "Therefore, all dogs are animals.")
INVALID = ("All dogs are mammals. Some cars are red. All cats are mammals. No rivers are mountains. "
           "Therefore, all dogs are cats.")
TIE = ("All dogs are loyal pets. All loyal pets are animals. All dogs are mammals. All mammals are animals. "
       "Some cars are red. Therefore, all dogs are animals.")
UNDECIDED = "Dogs bark loudly. Cars go fast. Rivers flow downhill. Mountains stand tall. Therefore, dogs run."


# ---------------------------------------------------------------- parser (ST1) použitý na dvojice
def test_parser_symmetric_on_premise_swap():
    """S2 skúša každú dvojicu len v jednom poradí - parser preto nesmie závisieť od poradia premís."""
    for x in TRAIN + DEV:
        s = split_sentences(x["syllogism"])
        a, b = parse_syllogism(" ".join(s)), parse_syllogism(" ".join([s[1], s[0], s[2]]))
        assert a.ok == b.ok, x["id"]
        if a.ok:
            assert is_valid_stmts(a.stmts) == is_valid_stmts(b.stmts), x["id"]


# ---------------------------------------------------------------- generátor dev množín
@pytest.mark.parametrize("variant", VARIANTS)
def test_devset_invariants(variant):
    base = DEV[:40]
    data = make_st2_dev(base, TRAIN, variant, seed=7)
    assert len(data) == len(base)
    train_premises = {s for x in TRAIN for s in split_sentences(x["syllogism"])[:2]}
    for x, src in zip(data, base):
        orig = split_sentences(src["syllogism"])
        sents = split_sentences(x["syllogism"])
        prem = sents[:-1]
        assert sents[-1] == orig[2]                                   # záver ostáva posledný
        assert 5 <= len(prem) <= 7
        assert sorted(prem[i] for i in x["orig_premises"]) == sorted(orig[:2])
        assert x["relevant_premises"] == (x["orig_premises"] if src["validity"] else [])
        assert x["validity"] == src["validity"] and x["plausibility"] == src["plausibility"]
        dist = [p for i, p in enumerate(prem) if i not in x["orig_premises"]]
        assert all(p in train_premises for p in dist)                 # rušivé premisy iba z train


def test_devset_deterministic_and_legacy_interface():
    a = make_st2_dev(DEV[:30], TRAIN, "mixed", 3)
    assert a == make_st2_dev(DEV[:30], TRAIN, "mixed", 3)
    assert a == make_st2_like(DEV[:30], TRAIN, seed=3, hard=True)
    assert make_st2_dev(DEV[:30], TRAIN, "easy", 3) == make_st2_like(DEV[:30], TRAIN, seed=3, hard=False)
    assert a != make_st2_dev(DEV[:30], TRAIN, "mixed", 4)


def test_hard_variant_has_more_conclusion_sharing_distractors():
    k = lambda v: np.mean([x["n_distractors_sharing_conclusion"] for x in make_st2_dev(DEV, TRAIN, v, 1)])
    assert k("easy") < k("mixed") < k("hard")


# ---------------------------------------------------------------- symbolický rozhodovač S2
def test_solve_valid_invalid_undecided():
    d = solve(VALID)
    assert (d.validity, d.premises, d.status) == (True, [0, 2], "valid-pair")
    d = solve(INVALID)
    assert (d.validity, d.premises, d.status) == (False, [], "invalid-sure")
    d = solve(UNDECIDED)
    assert d.validity is None and d.status == "undecided" and d.reason == "conclusion-untyped"


def test_tie_breaks():
    """Dve platné dvojice: (0,1) s dlhším stredným termínom a (2,3) s užším prekryvom so záverom."""
    assert solve(TIE, "lowest").premises == [0, 1]
    assert solve(TIE, "jaccard").premises == [2, 3]
    assert solve(TIE, "struct").premises == [2, 3]
    assert solve(TIE, "union").premises == [0, 1, 2, 3]
    assert solve(TIE).tie and not solve(VALID).tie


def test_predict_fallbacks():
    data = [{"id": "u", "syllogism": UNDECIDED}, {"id": "v", "syllogism": VALID}]
    p, _ = predict(data, "invalid")
    assert p[0] == {"id": "u", "validity": False, "relevant_premises": []}
    p, _ = predict(data, "valid")
    assert p[0]["validity"] is True and p[0]["relevant_premises"] == overlap_pair(UNDECIDED)
    other = {"u": {"id": "u", "validity": True, "relevant_premises": [1, 3]}, "v": {"validity": False}}
    p, _ = predict(data, other)
    assert p[0]["relevant_premises"] == [1, 3]
    assert p[1] == {"id": "v", "validity": True, "relevant_premises": [0, 2]}   # rozhodnuté symbolicky, fallback sa nepoužije


def test_overlap_pair_returns_two_distinct_indices():
    for x in make_st2_dev(DEV[:20], TRAIN, "hard", 0):
        pr = overlap_pair(x["syllogism"])
        assert len(set(pr)) == 2 and all(0 <= i < len(split_sentences(x["syllogism"])) - 1 for i in pr)


# ---------------------------------------------------------------- metrika ST2
def _random_preds(data, seed):
    rng = random.Random(seed)
    out = []
    for x in data:
        v = rng.random() < 0.5
        n = len(split_sentences(x["syllogism"])) - 1
        out.append({"id": x["id"], "validity": v, "relevant_premises": sorted(rng.sample(range(n), rng.choice([1, 2, 3]))) if v else []})
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_st2_metric_matches_official(seed):
    data = make_st2_dev(DEV, TRAIN, "mixed", seed)
    pred = _random_preds(data, seed)
    off = official_scores_st2(data, pred)
    fast = score_arrays_st2(*st2_arrays(data, pred))
    for k in ("accuracy", "f1_premises", "tce", "combined_score"):
        assert fast[k] == pytest.approx(off[k], abs=1e-9), k


def test_s2_output_passes_official_file_scoring():
    data = make_st2_dev(DEV, TRAIN, "mixed", 42)
    pred, dec = predict(data, "invalid", "struct", "any_parsed")
    with tempfile.TemporaryDirectory() as d:
        g, p, o = (os.path.join(d, n) for n in ("gold.json", "pred.json", "out.json"))
        for path, obj in ((g, data), (p, pred)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        official_24.run_full_scoring(g, p, o)
        with open(o, encoding="utf-8") as f:
            res = json.load(f)
    ours = st2_breakdown(data, pred, dec)
    for k, ko in (("accuracy", "accuracy"), ("f1_premises", "f1_premises"), ("tce", "content_effect"),
                  ("combined_score", "combined_score")):
        assert res[ko] == pytest.approx(ours[k], abs=1e-3), k
    assert sum(b["n"] for b in ours["branches"].values()) == len(data)


# ---------------------------------------------------------------- N1 pre ST2: abstrakcia, prompt, extrakcia odpovede
from src.st2_abstraction import abstract_lexical, content_lemmas, PSEUDOWORDS
from src.neurosymbolic import PROMPT_V1
from src import st2_llm
from src.st2_llm import PROMPT_ST2_V1, format_argument, build_messages, extract_answer, to_prediction, text_sha
from src.st2_encoder import build_pair_examples, all_pairs, pair_text, scores_to_predictions, mean_scores


def test_abstraction_preserves_parser_decision_on_st1():
    """Abstrakcia nesmie zmeniť štruktúru, ktorú vidí parser: rovnaké rozparsovanie aj platnosť na ST1 train+dev."""
    for x in TRAIN + DEV:
        a, b = parse_syllogism(x["syllogism"]), parse_syllogism(abstract_lexical(x["syllogism"]))
        assert a.ok == b.ok, x["id"]
        if a.ok:
            assert is_valid_stmts(a.stmts) == is_valid_stmts(b.stmts), x["id"]


def test_abstraction_hides_content_and_is_consistent():
    t = abstract_lexical(VALID)
    assert not ({"dog", "mammal", "animal", "car", "river"} & content_lemmas(t))
    assert len(split_sentences(t)) == len(split_sentences(VALID))
    w = t.split()[1]                                      # "dogs" v 1. premise a v závere -> rovnaké umelé slovo
    assert t.count(w) == 2 and w.rstrip("s") in PSEUDOWORDS


def test_st2_prompt_extends_st1_prompt_and_numbers_from_one():
    assert PROMPT_ST2_V1["system"].startswith(PROMPT_V1["system"])
    arg = format_argument(VALID)
    assert arg.splitlines()[0].startswith("Premise 1: ") and arg.splitlines()[-1].startswith("Conclusion: Therefore")
    assert len(arg.splitlines()) == 5
    m = build_messages(VALID, "abstract")
    assert [r["role"] for r in m] == ["system", "user"] and "dog" not in m[1]["content"].lower()


@pytest.mark.parametrize("raw,n,exp", [
    ("VALID 2 5", 6, (True, [1, 4], True)),
    ("INVALID", 6, (False, [], True)),
    ("The argument is valid: premises 3 and 1.", 5, (True, [0, 2], True)),
    ("VALID 1 9 4", 5, (True, [0, 3], True)),             # 9 je mimo rozsahu -> preskočí sa
    ("VALID 3", 5, (True, [2], False)),
    ("I cannot tell.", 5, (None, [], False)),
    ("INVALID, not VALID 1 2", 5, (False, [], True)),      # rozhoduje prvý verdikt
])
def test_extract_answer(raw, n, exp):
    a = extract_answer(raw, n)
    assert (a["validity"], a["premises"], a["ok"]) == exp


def test_to_prediction_unparsable_stays_none_like_st1():
    """Ako N1 v ST1: nejednoznačná odpoveď ostáva None (abstencia), nikdy sa automaticky nepočíta ako INVALID."""
    x = {"id": "q", "syllogism": VALID}
    p, f = to_prediction(x, "hmm")
    assert p == {"id": "q", "validity": None, "relevant_premises": []} and f["unparsable"]


def _fake_n1(monkeypatch, tmp_path, answers):
    """N1 pre ST2 s falošným runnerom (ako Samov test run_llama): bez GPU a bez sťahovania modelu."""
    calls = []
    class FakeRunner:
        revision = "frozen-test-revision"
        def __init__(self, model, revision, quantize=True):
            calls.append(("init", revision, quantize))
    def fake_generate(runner, messages, max_new_tokens, item_id=""):
        calls.append(("gen", item_id)); return answers.get(item_id, "INVALID")
    sets = {"dev_x_s1": [{"id": "a", "syllogism": VALID}, {"id": "b", "syllogism": INVALID}]}
    monkeypatch.setattr(st2_llm, "ROOT", tmp_path)
    monkeypatch.setattr(st2_llm, "LlamaRunner", FakeRunner)
    monkeypatch.setattr(st2_llm, "generate", fake_generate)
    monkeypatch.setattr(st2_llm, "eval_sets", lambda cfg, split: sets)
    return calls, sets


def test_n1_st2_journal_resume_and_identity(tmp_path, monkeypatch):
    calls, sets = _fake_n1(monkeypatch, tmp_path, {"a": "VALID 1 3"})
    st2_llm.main(["--representations", "abstract", "--output", "out"])
    assert [c for c in calls if c[0] == "gen"] == [("gen", "a"), ("gen", "b")]
    assert calls[0] == ("init", st2_llm.parser().parse_args([]).revision, True)      # 4-bit + revízia ako ST1
    st2_llm.main(["--representations", "abstract", "--output", "out"])               # pokračovanie z journalu
    assert len([c for c in calls if c[0] == "gen"]) == 2
    p, info = st2_llm.load_predictions("dev_x_s1", sets["dev_x_s1"], "abstract", "out")
    assert p[0]["relevant_premises"] == [0, 2] and p[1]["validity"] is False and info["unparsable_answers"] == 0
    with pytest.raises(ValueError):                                                     # iná konfigurácia behu
        st2_llm.main(["--representations", "abstract", "--output", "out", "--max-new-tokens", "9"])
    bad = [dict(sets["dev_x_s1"][0], syllogism=INVALID), sets["dev_x_s1"][1]]
    with pytest.raises(ValueError):                                                     # iná verzia množiny
        st2_llm.load_predictions("dev_x_s1", bad, "abstract", "out")
    assert st2_llm.load_predictions("dev_x_s1", sets["dev_x_s1"], "original", "out") is None


def test_st2_neural_settings_match_st1_runs():
    """Revízie a hyperparametre ST2 = to, čo Samo reálne použil v ST1 (jeho uložené run.json / training.json)."""
    from src.st2_encoder import parser as st2_enc_parser
    from src.finetune_encoder import parser as st1_enc_parser
    llama_run = load_json("outputs/neural/llama/dev/abstract/run.json")
    n1 = st2_llm.parser().parse_args([])
    assert (n1.model, n1.revision, n1.split, n1.no_4bit) == (llama_run["model"], llama_run["revision"], "dev", False)
    assert llama_run["quantization"] == "nf4-double"
    b2_run = load_json("outputs/neural/b2/seed_0/training.json")
    st2, st1 = vars(st2_enc_parser().parse_args([])), vars(st1_enc_parser().parse_args([]))
    for k in ("model", "seeds", "epochs", "lr", "batch", "accumulation", "max_length", "mode"):
        assert st2[k] == st1[k], k
    assert st2["revision"] == b2_run["revision"] and b2_run["max_length"] == st2["max_length"]


def test_abstention_withholds_official_metrics_and_propagates_through_hybrid():
    data = [{"id": "u", "syllogism": UNDECIDED, "validity": True, "plausibility": True, "relevant_premises": [0, 1]},
            {"id": "v", "syllogism": VALID, "validity": True, "plausibility": False, "relevant_premises": [0, 2]}]
    llm = {"u": {"id": "u", "validity": None, "relevant_premises": []},
           "v": {"id": "v", "validity": True, "relevant_premises": [0, 2]}}
    p, dec = predict(data, llm)
    assert p[0]["validity"] is None and p[1]["validity"] is True       # hybrid abstinuje iba tam, kde LLM
    r = st2_breakdown(data, p, dec)
    assert r["official_withheld"] and r["answer_coverage"] == 0.5 and r["accuracy_with_abstentions_as_errors"] == 50


# ---------------------------------------------------------------- B2 pre ST2: párové dáta a predikcie
def test_pair_examples_labels():
    data = make_st2_dev(TRAIN[:60], TRAIN, "mixed", 0)
    ex = build_pair_examples(data, neg_per_example=4, seed=0)
    by_id = {}
    for e in ex:
        by_id.setdefault(e["id"], []).append(e)
    for x in data:
        rows = by_id[x["id"]]
        gold = sorted(x["orig_premises"])
        assert [r for r in rows if r["pair"] == gold][0]["label"] == int(x["validity"])   # neplatná dvojica = ST1 negatív
        assert sum(r["label"] for r in rows) == int(x["validity"])
        assert len(rows) == 5 and len({tuple(r["pair"]) for r in rows}) == 5
        s = split_sentences(x["syllogism"])
        assert all(r["text"] == pair_text(s[:-1], *r["pair"], s[-1]) for r in rows)


def test_scores_to_predictions_and_mean():
    x = {"id": "a", "syllogism": VALID}
    assert len(all_pairs(x)) == 6
    sc = {"a": {"pairs": [[0, 1, 0.2], [0, 2, 0.9], [1, 3, 0.4]]}}
    assert scores_to_predictions([x], sc) == [{"id": "a", "validity": True, "relevant_premises": [0, 2]}]
    assert scores_to_predictions([x], sc, threshold=0.95)[0] == {"id": "a", "validity": False, "relevant_premises": []}
    m = mean_scores([sc, {"a": {"pairs": [[0, 1, 0.8], [0, 2, 0.1], [1, 3, 0.4]]}}])
    assert m["a"]["pairs"][0][2] == pytest.approx(0.5)


# ---------------------------------------------------------------- voľba zálohy S2 (iba dev, vopred zapísané pravidlo)
def test_select_fallback_rule():
    from src.run_st2 import select_fallback
    cfg = {"st2": {"symbolic": {"tie_break": "struct", "invalid_rule": "any_parsed", "fallback": "B1"},
                   "fallback_selection": {"candidates": ["invalid", "B1", "N1-abstract"], "tie_tolerance": 0.5}}}
    row = lambda *cs: {v: {"combined_score": c} for v, c in zip(("easy", "mixed", "hard"), cs)}
    name = lambda fb: f"S2 tie=struct inv=any_parsed fb={fb}"
    s = {name("invalid"): row(26, 26, 26), name("B1"): row(28, 28, 28)}
    r = select_fallback(s, cfg)
    assert r["selected"] == "B1" and r["missing"] == ["N1-abstract"] and r["matches_config"]
    s[name("N1-abstract")] = row(28.3, 28.3, 28.3)          # lepší o menej než tolerancia -> ostáva skorší v zozname
    assert select_fallback(s, cfg)["selected"] == "B1"
    s[name("N1-abstract")] = {v: {"official_withheld": True} for v in ("easy", "mixed", "hard")}
    r = select_fallback(s, cfg)                         # abstencie -> vylúčený, nie chýbajúci (test nezablokuje)
    assert r["excluded"] == ["N1-abstract"] and not r["missing"] and r["selected"] == "B1"
    s[name("N1-abstract")] = row(31, 31, 31)
    r = select_fallback(s, cfg)
    assert r["selected"] == "N1-abstract" and not r["matches_config"] and not r["missing"]
