"""Neuro-symbolický baseline N1: regex parser -> abstrakcia (X, Y, Z) -> zero-shot LLM.

* Symbolická časť (parser, abstrakcia, tabuľka platnosti) je plne deterministická a beží lokálne (`src/symbolic.py`).
* LLM časť (`run_llm`) vyžaduje GPU a prístup k modelu (Hugging Face). V tomto prostredí NEBOLA spustená;
  funkcia je pripravená pre Colab/GPU a pred použitím výsledkov sa musí spustiť a overiť.
"""
from __future__ import annotations
import hashlib, json, re
from .symbolic import parse_syllogism, abstract_text, canonical_text, is_valid_stmts

PROMPT_V1 = {
    "system": ("You are a careful logician. You judge only the FORM of an argument, never whether its statements are "
               "true in the real world. The symbols X, Y and Z (or any words) stand for arbitrary sets. Every set is non-empty "
               "(existential import). 'All A are B' means every member of A is in B; 'No A are B' means A and B share no member; "
               "'Some A are B' means at least one member lies in both; 'Some A are not B' means at least one member of A is outside B. "
               "An argument is VALID if the conclusion must be true whenever both premises are true; otherwise it is INVALID."),
    "user": "{argument}\n\nIs this argument formally valid? Answer with exactly one word: VALID or INVALID.",
}
PROMPT_SHA = hashlib.sha256(json.dumps(PROMPT_V1, sort_keys=True).encode()).hexdigest()[:12]


def build_item(x: dict) -> dict:
    """Spracuje jeden príklad: rozparsuje a pripraví všetky reprezentácie vstupu pre LLM."""
    p = parse_syllogism(x["syllogism"])
    item = {"id": x["id"], "original": x["syllogism"], "parsed": p.ok, "parse_failure": p.reason or None}
    if p.ok:
        item.update(abstract=abstract_text(x["syllogism"], p), canonical=canonical_text(p),
                    statements=[list(s) for s in p.stmts], symbolic_valid=is_valid_stmts(p.stmts))
    return item


def argument_text(item: dict, representation: str, fallback: str = "original") -> str:
    rep = representation if item["parsed"] else fallback
    return item[rep]


def make_messages(item: dict, representation: str = "abstract", fallback: str = "original") -> list[dict]:
    return [{"role": "system", "content": PROMPT_V1["system"]},
            {"role": "user", "content": PROMPT_V1["user"].format(argument=argument_text(item, representation, fallback))}]


_ANS = re.compile(r"\b(invalid|valid)\b", re.I)


def extract_label(output: str) -> bool | None:
    """VALID -> True, INVALID -> False; nejednoznačný výstup -> None (počíta sa ako chyba, nie hádanie)."""
    m = _ANS.findall(output)
    if not m or len({w.lower() for w in m}) != 1:
        return None
    return m[0].lower() == "valid"


def run_llm(items: list[dict], model_id: str, representation: str, fallback: str = "original",
            max_new_tokens: int = 8) -> list[dict]:   # pragma: no cover - vyžaduje GPU
    """Compatibility wrapper; use python -m src.run_llama for saved experiments."""
    from .run_llama import LlamaRunner
    runner = LlamaRunner(model_id, 'main', quantize=False)
    return [runner.predict(it, representation, max_new_tokens, fallback) for it in items]
