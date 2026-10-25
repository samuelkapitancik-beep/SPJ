"""N1 pre ST2: parser -> abstrakcia -> zero-shot Llama 3 8B, rovnaké načítanie modelu ako N1 v ST1.

Model sa načítava IMPORTOVANÝM `run_llama.LlamaRunner` (NF4 4-bit + double quant, compute dtype bf16/fp16 podľa GPU,
pevná revízia z configs/st2.yaml = revízia N1 v ST1). Generovanie je rovnaké ako `LlamaRunner.predict` (po jednom
príklade, greedy, stop na EOS alebo <|eot_id|>, odmietnutie príliš dlhého vstupu); `predict` sa priamo volať nedá,
lebo má natvrdo prompt ST1 a vracia iba platnosť. Pokračovanie po prerušení: `run_llama.cached_predictions`
+ kontrola identity behu (run.json), ako v ST1.

Rozdiely oproti N1 v ST1 (zdôvodnenie v správe):
  * abstrakcia `st2_abstraction.abstract_lexical` (obsahové lemmy -> umelé slová) namiesto X/Y/Z: X/Y/Z vyžaduje úspešný
    parse jedného sylogizmu, v ST2 je termínov viac (rušivé vety) a LLM je potrebné práve tam, kde parser zlyhal;
  * výstup "VALID i j" / "INVALID" (premisy číslované od 1) -> max_new_tokens 16 namiesto 8 (dve čísla navyše);
  * reprezentácie `abstract` (N1) a `original` (rovnaký prompt na pôvodnom texte = zero-shot ablácia).
Nejednoznačná odpoveď ostáva `null` (nikdy sa automaticky nepočíta ako INVALID), rovnako ako v ST1.

Colab (GPU):  python -m src.st2_llm --split dev     # 9 syntetických dev množín
              python -m src.st2_llm --split test    # test ST2: iba generovanie, NEHODNOTÍ sa (to robí run_st2 --final)
Lokálne:      `load_predictions` (bez torch) číta journal pre src/run_st2.py.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, re
from pathlib import Path
from .utils import ROOT, save_json, set_seed
from .data_loader import load_json
from .preprocessing import split_sentences
from .st2_abstraction import abstract_lexical
from .st2_config import load_st2_config
from .neurosymbolic import PROMPT_V1    # iba konštanta promptu ST1 (rovnaký štýl promptu)
from .run_llama import LlamaRunner, cached_predictions   # načítanie modelu presne ako N1 v ST1
from .neural_common import environment, fingerprint

REPRESENTATIONS = ("abstract", "original")

PROMPT_ST2_V1 = {
    "system": PROMPT_V1["system"] + (" Some of the listed premises may be irrelevant to the conclusion. The argument is "
                                     "VALID if two of the premises together make the conclusion necessarily true; "
                                     "otherwise it is INVALID."),
    "user": ("{argument}\n\nDo two of the premises make the conclusion formally valid? Answer with exactly one line: "
             "VALID followed by the numbers of the two premises (for example: VALID 2 5), or INVALID."),
}
PROMPT_ST2_SHA = hashlib.sha256(json.dumps(PROMPT_ST2_V1, sort_keys=True).encode()).hexdigest()[:12]


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def format_argument(text: str) -> str:
    """Premisy očíslované od 1, záver zvlášť (posledná veta)."""
    sents = split_sentences(text)
    lines = [f"Premise {k}: {s}" for k, s in enumerate(sents[:-1], start=1)]
    return "\n".join(lines + [f"Conclusion: {sents[-1]}"])


def build_messages(text: str, representation: str) -> list[dict]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"neznáma reprezentácia {representation!r}, povolené {REPRESENTATIONS}")
    arg = format_argument(abstract_lexical(text) if representation == "abstract" else text)
    return [{"role": "system", "content": PROMPT_ST2_V1["system"]},
            {"role": "user", "content": PROMPT_ST2_V1["user"].format(argument=arg)}]


_VERDICT = re.compile(r"\b(invalid|valid)\b", re.I)


def extract_answer(output: str, n_premises: int) -> dict:
    """Prvý verdikt v odpovedi; pri VALID prvé dve rôzne čísla 1..n za ním (-> indexy od 0).
    Chýbajúci verdikt -> validity None (abstencia, nepočíta sa automaticky ako INVALID)."""
    m = _VERDICT.search(output)
    if not m:
        return {"validity": None, "premises": [], "ok": False}
    if m.group(1).lower() == "invalid":
        return {"validity": False, "premises": [], "ok": True}
    nums = []
    for k in re.findall(r"\d+", output[m.end():]):
        k = int(k)
        if 1 <= k <= n_premises and k - 1 not in nums:
            nums.append(k - 1)
        if len(nums) == 2:
            break
    return {"validity": True, "premises": sorted(nums), "ok": len(nums) == 2}


def to_prediction(x: dict, raw: str) -> tuple[dict, dict]:
    """Surová odpoveď -> predikcia ST2 (validity True/False/None) + príznaky pre diagnostiku."""
    a = extract_answer(raw, len(split_sentences(x["syllogism"])) - 1)
    v = a["validity"]
    return ({"id": x["id"], "validity": v, "relevant_premises": a["premises"] if v else []},
            {"unparsable": v is None, "bad_premises": v is True and not a["ok"]})


# ------------------------------------------------------------------------------------------------ model (Colab)
def generate(runner, messages: list[dict], max_new_tokens: int, item_id: str = "") -> str:   # pragma: no cover - GPU
    """Rovnaké generovanie ako `LlamaRunner.predict` (ST1), iba so zadanými správami."""
    tok, model, torch = runner.tokenizer, runner.model, runner.torch
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(f"Argument exceeds context window: {item_id}")      # bez orezania premís
    eos = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tok.unk_token_id and eot not in eos:
        eos.append(eot)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             eos_token_id=eos, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------------------------------------ journal
def run_dir(output: str, set_name: str, representation: str) -> Path:
    return ROOT / output / set_name / representation


def load_predictions(set_name: str, data: list[dict], representation: str, output: str):
    """Lokálne: predikcie N1 z journalu. Vráti (predikcie, štatistiky) alebo None, ak beh chýba alebo je neúplný.
    Kontroluje poradie id (ako ST1) aj hash textu (iná verzia množiny -> chyba)."""
    path = run_dir(output, set_name, representation) / "responses.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < len(data):
        return None
    if [r["id"] for r in rows] != [x["id"] for x in data]:
        raise ValueError(f"N1 {set_name}/{representation}: poradie alebo id v journale nesedí s množinou")
    preds, n_unp, n_bad = [], 0, 0
    for x, r in zip(data, rows):
        if r["text_sha"] != text_sha(x["syllogism"]):
            raise ValueError(f"N1 {set_name}/{representation}: text príkladu {x['id']} nesedí (iná verzia množiny?)")
        p, flags = to_prediction(x, r["raw"])
        preds.append(p); n_unp += flags["unparsable"]; n_bad += flags["bad_premises"]
    return preds, {"unparsable_answers": n_unp, "valid_with_bad_premises": n_bad}


def eval_sets(cfg: dict, split: str) -> dict[str, list[dict]]:
    """Množiny pre generovanie: 'dev' = 9 dev množín (st2.dev_dir), 'test' = test ST2 (pod menom 'test')."""
    if split == "test":
        return {"test": load_json(cfg["paths"]["test_st2"])}
    files = sorted(glob.glob(str(ROOT / cfg["st2"]["dev_dir"] / "dev_*_s*.json")))
    if not files:
        raise SystemExit("Chýbajú dev množiny v data/processed/st2 - najprv spusti lokálne `python -m src.run_st2`.")
    return {Path(f).stem: load_json(f) for f in files}


def parser():
    n1 = load_st2_config()["st2"]["n1"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=n1["model_id"])
    p.add_argument("--revision", default=n1["revision"], help="predvolene revízia N1 v ST1")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    p.add_argument("--max-new-tokens", type=int, default=n1["max_new_tokens"])
    p.add_argument("--no-4bit", action="store_true", help="iba ak by tak bežal aj N1 v ST1 (porovnateľnosť)")
    p.add_argument("--output", default=n1["output"])
    p.add_argument("--limit", type=int, default=None, help="skúška: iba N príkladov, výstup do <output>_skuska")
    return p


def main(argv=None):   # pragma: no cover - vyžaduje GPU (overené testom s falošným runnerom)
    a = parser().parse_args(argv)
    if a.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive.")
    cfg = load_st2_config()
    set_seed(cfg["seed"])
    output = a.output + ("_skuska" if a.limit else "")
    sets = {k: (v[: a.limit] if a.limit else v) for k, v in eval_sets(cfg, a.split).items()}
    runner = LlamaRunner(a.model, a.revision, quantize=not a.no_4bit)
    for name, data in sets.items():
        for rep in a.representations:
            dest = run_dir(output, name, rep)
            dest.mkdir(parents=True, exist_ok=True)
            identity = {"experiment": "N1-ST2", "model": a.model, "revision": runner.revision, "set": name,
                        "representation": rep, "prompt_version": "st2_v1", "prompt_sha256_12": PROMPT_ST2_SHA,
                        "prompt": PROMPT_ST2_V1, "items_sha256": fingerprint(data),
                        "quantization": "nf4-double" if not a.no_4bit else "none",
                        "max_new_tokens": a.max_new_tokens, "do_sample": False, "seed": cfg["seed"],
                        "environment": environment()}
            identity_path = dest / "run.json"
            if identity_path.exists() and load_json(identity_path) != identity:
                raise ValueError(f"Run configuration changed. Use a new --output directory: {dest}")
            save_json(identity, identity_path)
            journal = dest / "responses.jsonl"
            done = cached_predictions(journal, data)
            with journal.open("a", encoding="utf-8") as stream:
                for k, x in enumerate(data[len(done):], start=len(done) + 1):
                    raw = generate(runner, build_messages(x["syllogism"], rep), a.max_new_tokens, x["id"])
                    stream.write(json.dumps({"id": x["id"], "text_sha": text_sha(x["syllogism"]), "raw": raw},
                                            ensure_ascii=False) + "\n")
                    stream.flush()
                    if k % 20 == 0:
                        print(f"{name}/{rep}: {k}/{len(data)}", flush=True)


if __name__ == "__main__":
    main()
