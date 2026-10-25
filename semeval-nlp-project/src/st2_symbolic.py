"""Subtask 2 (ST2): symbolický výber relevantných premís a rozhodnutie o platnosti.

Vstup ST2 je sylogizmus s 5–7 premisami, z ktorých nanajvýš dve sú relevantné; posledná veta je záver.
Postup (S2, pôvodný návrh M. Kelemen):
  1. rozdeliť text na vety, posledná je záver;
  2. pre každú dvojicu premís (i, j) zostaviť 3-vetný sylogizmus „premisa_i premisa_j záver“
     a rozparsovať ho existujúcim parserom zo ST1 (`symbolic.parse_syllogism`; je symetrický voči poradiu premís);
  3. ak je niektorá rozparsovaná dvojica podľa tabuľky platnosti platná -> validity = True,
     relevantné premisy = táto dvojica; pri viacerých platných dvojiciach rozhoduje pravidlo `tie_break`;
  4. ak sa rozparsovala aspoň jedna dvojica, ale žiadna nie je platná -> validity = False, [] (stav `invalid-sure`),
     pri `invalid_rule="strict"` však iba vtedy, ak sa dá otypovať každá premisa zdieľajúca obsahové slovo so záverom
     (inak môže byť nerozparsovaná práve relevantná premisa a „neplatné“ by bola chyba, ktorá stojí ACC aj F1);
  5. inak parser nevie rozhodnúť (stav `undecided`); rozhoduje záložná stratégia (`fallback`).

Pravidlá pre remízy (`tie_break`):
  * `lowest`  - najnižšie indexy (referencia),
  * `jaccard` - najväčší Jaccardov koeficient lemm (premisy dvojice vs. záver); pôvodná voľba,
  * `struct`  - najmenší symetrický rozdiel medzi termínmi X∪Y nájdenými v premisách a lemmami záveru
                (pri správnej dvojici sa termíny premís a záveru lexikálne zhodujú), pri zhode Jaccard, potom indexy,
  * `union`   - zjednotenie všetkých platných dvojíc. LEN ABLÁCIA: úloha žiada presne potrebné premisy,
                zjednotenie iba využíva čiastočný kredit makro-F1.

Parser ani tabuľka sa pre ST2 nemenili. Voľby (`tie_break`, `invalid_rule`, `fallback`) sa robia na syntetických
dev množinách (`st2_devset`), nie na teste.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from .preprocessing import split_sentences
from .symbolic import parse_syllogism, is_valid_stmts, statement_type, Parse
from .st2_abstraction import content_lemmas

TIE_BREAKS = ("lowest", "jaccard", "struct", "union")
INVALID_RULES = ("any_parsed", "strict")


@dataclass
class St2Decision:
    validity: bool | None                  # None = parser nevie rozhodnúť
    premises: list[int] = field(default_factory=list)
    status: str = ""                        # valid-pair | invalid-sure | undecided
    reason: str = ""                        # dôvod pri invalid-sure / undecided
    n_premises: int = 0
    parsed_pairs: list[tuple[int, int, bool]] = field(default_factory=list)   # (i, j, platná?)
    valid_pairs: list[tuple[int, int]] = field(default_factory=list)          # zoradené podľa tie_break

    @property
    def tie(self) -> bool:
        return len(self.valid_pairs) > 1


def split_premises(text: str) -> tuple[list[str], str]:
    sents = split_sentences(text)
    return sents[:-1], sents[-1]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def overlap_pair(text: str) -> list[int]:
    """Čisto lexikálna heuristika (bez parsera): dvojica premís s najväčším Jaccardom lemm voči záveru.
    Používa sa v baselinoch a v záložnej stratégii, keď parser nerozhodne."""
    premises, conclusion = split_premises(text)
    c = content_lemmas(conclusion)
    lem = [content_lemmas(p) for p in premises]
    pairs = list(combinations(range(len(premises)), 2))
    best = max(pairs, key=lambda ij: (jaccard(lem[ij[0]] | lem[ij[1]], c), -ij[0], -ij[1]))
    return list(best)


def _order_valid(valid: list[tuple[int, int]], parses: dict, premises: list[str], conclusion: str,
                 tie_break: str) -> list[tuple[int, int]]:
    c = content_lemmas(conclusion)
    jac = lambda ij: jaccard(content_lemmas(premises[ij[0]]) | content_lemmas(premises[ij[1]]), c)
    if tie_break in ("lowest", "union"):
        return sorted(valid)
    if tie_break == "jaccard":
        return sorted(valid, key=jac, reverse=True)           # stabilné: pri zhode ostáva poradie (i, j)
    if tie_break == "struct":
        def key(ij):
            p: Parse = parses[ij]
            xy = set(p.spans["X"]) | set(p.spans["Y"])
            return (len(xy ^ c), -jac(ij), ij)
        return sorted(valid, key=key)
    raise ValueError(f"neznáme tie_break {tie_break!r}, povolené {TIE_BREAKS}")


def solve(text: str, tie_break: str = "jaccard", invalid_rule: str = "any_parsed") -> St2Decision:
    """Symbolické rozhodnutie pre jeden ST2 príklad. Predvolené argumenty = pôvodné správanie S2."""
    if invalid_rule not in INVALID_RULES:
        raise ValueError(f"neznáme invalid_rule {invalid_rule!r}, povolené {INVALID_RULES}")
    premises, conclusion = split_premises(text)
    parsed, valid, parses = [], [], {}
    for i, j in combinations(range(len(premises)), 2):
        p = parse_syllogism(f"{premises[i]} {premises[j]} {conclusion}")
        if not p.ok:
            continue
        v = is_valid_stmts(p.stmts)
        parsed.append((i, j, v))
        if v:
            valid.append((i, j)); parses[(i, j)] = p
    d = St2Decision(None, [], "", "", len(premises), parsed, [])
    if valid:
        d.valid_pairs = _order_valid(valid, parses, premises, conclusion, tie_break)
        d.validity, d.status = True, "valid-pair"
        d.premises = (sorted({k for ij in d.valid_pairs for k in ij}) if tie_break == "union"
                      else list(d.valid_pairs[0]))
        return d
    if not parsed:
        d.status = "undecided"
        d.reason = "conclusion-untyped" if statement_type(conclusion)[0] is None else "no-pair-parsed"
        return d
    if invalid_rule == "strict":
        c = content_lemmas(conclusion)
        untyped = [i for i, s in enumerate(premises) if content_lemmas(s) & c and statement_type(s)[0] is None]
        if untyped:
            d.status, d.reason = "undecided", "untyped-premise-shares-conclusion-term"
            return d
    d.validity, d.status = False, "invalid-sure"
    return d


def predict(data: list[dict], fallback="invalid", tie_break: str = "jaccard", invalid_rule: str = "any_parsed"):
    """S2: predikcie v oficiálnom formáte ST2 + zoznam rozhodnutí (na diagnostiku).

    `fallback` pre stav `undecided`:
      * "invalid" (alebo False) - neplatné, [] (pôvodné správanie),
      * "valid"   (alebo True)  - platné, dvojica podľa `overlap_pair`,
      * dict id -> predikcia    - predikcia iného modelu (napr. B1 alebo neskôr LLM) v oficiálnom formáte.
    """
    preds, decisions = [], []
    for x in data:
        d = solve(x["syllogism"], tie_break, invalid_rule)
        if d.validity is not None:
            pr = {"id": x["id"], "validity": d.validity, "relevant_premises": d.premises if d.validity else []}
        elif isinstance(fallback, dict):
            f = fallback[x["id"]]
            v = f["validity"]                     # None = záloha (LLM) neodpovedala -> abstencia ostáva
            pr = {"id": x["id"], "validity": None if v is None else bool(v),
                  "relevant_premises": list(f["relevant_premises"]) if v else []}
        elif fallback in ("valid", True):
            pr = {"id": x["id"], "validity": True, "relevant_premises": overlap_pair(x["syllogism"])}
        elif fallback in ("invalid", False):
            pr = {"id": x["id"], "validity": False, "relevant_premises": []}
        else:
            raise ValueError(f"neznámy fallback {fallback!r}")
        preds.append(pr); decisions.append(d)
    return preds, decisions
