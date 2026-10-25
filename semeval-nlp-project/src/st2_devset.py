"""Syntetické vývojové množiny pre ST2 vytvorené z dát ST1 (augmentácia).

ST2 nemá trénovacie ani vývojové dáta, iba test. Aby sme metódy nevyvíjali na teste, vytvoríme z 3-vetných
sylogizmov ST1 príklady vo formáte ST2: k dvom pôvodným premisám pridáme rušivé premisy z iných príkladov
a všetko náhodne premiešame (záver ostáva posledný). Štítky:
  * validity a plausibility sa preberajú z pôvodného príkladu,
  * relevant_premises = nové pozície dvoch pôvodných premís, ak je sylogizmus platný; inak [] (ako v oficiálnych dátach).
Počet premís (5–7) sa losuje s rovnakým rozložením ako v teste ST2 (70 / 67 / 53) - z testu sa berie iba toto
rozloženie dĺžok, nie obsah.

Základ je `make_st2_like` (M. Kelemen, pôvodne `st2_augment.py`). Varianty rušivých premís:
  * `easy`  - všetky rušivé premisy náhodne z poolu (spravidla iná téma, výber premís je ľahký),
  * `mixed` - približne polovica rušivých premís zdieľa obsahové slovo s pôvodným sylogizmom, zvyšok náhodne
              (= pôvodné `make_st2_like(..., hard=True)`; rovnaký seed dáva rovnaké príklady),
  * `hard`  - rušivé premisy prednostne zdieľajú obsahové slovo so ZÁVEROM (doplnené náhodnými, ak ich je málo).
              Práve takéto vety vedú k falošným platným dvojiciam pri symbolickom výbere.
Proporciu ťažkých rušivých viet v teste ST2 neodhadujeme (test nepozeráme); varianty dávajú rozsah.

Rušivé premisy sa berú iba z `pool` (train), báza je dev. Pravidlá parsera boli ladené na train+dev, preto je
symbolická vetva na týchto množinách optimistická - nezávislé číslo dá až test ST2.
"""
from __future__ import annotations
import random
from .preprocessing import split_sentences
from .st2_abstraction import content_lemmas

N_PREMISES_WEIGHTS = {5: 70, 6: 67, 7: 53}      # rozloženie počtu premís v teste ST2
VARIANTS = ("easy", "mixed", "hard")


def _pool(pool: list[dict]):
    prem = [(x["id"], s) for x in pool for s in split_sentences(x["syllogism"])[:2]]
    return prem, [content_lemmas(s) for _, s in prem]


def make_st2_dev(base: list[dict], pool: list[dict], variant: str = "mixed", seed: int = 42) -> list[dict]:
    """Vytvorí ST2-podobnú množinu z `base` s rušivými premisami z `pool`. Deterministické pre daný seed."""
    if variant not in VARIANTS:
        raise ValueError(f"neznámy variant {variant!r}, povolené {VARIANTS}")
    rng = random.Random(seed)
    pool_prem, pool_lem = _pool(pool)
    sizes, weights = zip(*N_PREMISES_WEIGHTS.items())
    out = []
    for x in base:
        sents = split_sentences(x["syllogism"])
        if len(sents) != 3:
            continue
        p1, p2, concl = sents
        n = rng.choices(sizes, weights)[0]
        own = content_lemmas(p1) | content_lemmas(p2) | content_lemmas(concl)
        cand = [k for k, (pid, _) in enumerate(pool_prem) if pid != x["id"]]
        if variant == "mixed":
            near = [k for k in cand if pool_lem[k] & own]
            rng.shuffle(near)
            near_set = set(near)
            rest = [k for k in cand if k not in near_set]
            rng.shuffle(rest)
            k_near = min(len(near), (n - 2 + 1) // 2)        # približne polovica z rovnakej témy
            chosen = near[:k_near] + rest[: n - 2 - k_near]
        elif variant == "hard":
            cl = content_lemmas(concl)
            near = [k for k in cand if pool_lem[k] & cl]
            rng.shuffle(near)
            near_set = set(near)
            rest = [k for k in cand if k not in near_set]
            rng.shuffle(rest)
            k_near = min(len(near), n - 2)
            chosen = near[:k_near] + rest[: n - 2 - k_near]
        else:
            chosen = rng.sample(cand, n - 2)
        prem = [("orig", p1), ("orig", p2)] + [("dist", pool_prem[k][1]) for k in chosen]
        rng.shuffle(prem)
        relevant = [i for i, (kind, _) in enumerate(prem) if kind == "orig"] if x["validity"] else []
        out.append({
            "id": f"aug-{x['id']}",
            "syllogism": " ".join(s for _, s in prem) + " " + concl,
            "validity": x["validity"],
            "plausibility": x["plausibility"],
            "relevant_premises": relevant,
            # diagnostika (oficiálny skript ju ignoruje): pozície pôvodných premís aj pri neplatných príkladoch
            "orig_premises": [i for i, (kind, _) in enumerate(prem) if kind == "orig"],
            "n_distractors_sharing_conclusion": sum(1 for k in chosen if pool_lem[k] & content_lemmas(concl)),
        })
    return out


def make_st2_like(base: list[dict], pool: list[dict], seed: int = 42, hard: bool = True) -> list[dict]:
    """Pôvodné rozhranie (`st2_augment.py`): hard=True zodpovedá variantu `mixed`, hard=False variantu `easy`."""
    return make_st2_dev(base, pool, "mixed" if hard else "easy", seed)
