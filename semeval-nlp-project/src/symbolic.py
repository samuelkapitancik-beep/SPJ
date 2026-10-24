"""Vlastný nízkoúrovňový regex tokenizér a deterministický parser kategorických sylogizmov (angličtina).

Zodpovednosti:
  1. `tokenize`           - regex tokenizér (bez externých knižníc),
  2. `statement_type`     - typ výroku A/E/I/O z kvantifikátorov a negácie,
  3. `parse_syllogism`    - extrakcia troch termínov (X = podmet záveru, Y = prísudok záveru, Z = stredný termín),
  4. `abstract_text`      - nahradenie obsahových termínov symbolmi X/Y/Z v pôvodnom znení viet,
  5. `canonical_text`     - normalizovaný zápis (napr. "All Z are Y."),
  6. `is_valid`           - rozhodnutie o platnosti úplným prehľadom modelov (Vennove oblasti, existenčný predpoklad).

Parser je deterministický (žiadna náhodnosť, žiadny model) a v prípade neistoty vracia dôvod zlyhania namiesto hádania.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from itertools import product
from .preprocessing import split_sentences

# ----------------------------------------------------------------------------------------------- 1. tokenizér
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+|[^\w\s]")   # Unicode písmená (nie iba ASCII)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_IRREG = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse",
          "wolves": "wolf", "leaves": "leaf", "geese": "goose", "feet": "foot", "teeth": "tooth", "lives": "life",
          "knives": "knife", "has": "have", "does": "do", "is": "be", "are": "be"}


def lemma(tok: str) -> str:
    """Hrubý kľúč termínu: zjednotí jednotné/množné číslo a 3. osobu (dostatočné na párovanie termínov medzi vetami)."""
    if tok in _IRREG:
        return _IRREG[tok]
    t = tok
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif t.endswith(("sses", "xes", "ches", "shes")):
        t = t[:-2]
    elif t.endswith("ss") or len(t) <= 3:
        pass
    elif t.endswith("s"):
        t = t[:-1]
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


# slová, ktoré nenesú obsah termínu (kvantifikátory, spona, rámcové frázy generátora)
_STOP = set("""
a an the all every each any some no not none nothing everything anything whatever whoever there exist exists
is are be was were been being am do does did can cannot could would should will may might must it its this that these those
which who whom whose also too even only just indeed truly really certainly surely definitely absolutely
one ones thing things individual individuals entity entities item items portion subset subsets set sets
part parts few select certain several least most many
type types kind kinds form forms member members category categories class classes group groups sort sorts
considered classified classify classifying known called named labeled referred described regarded counted
case true fact false follows follow therefore thus hence so consequently accordingly then
logically necessarily necessitates implies imply means mean conclude conclusion proves prove shows show
result results lead leads led allows allow us we can one
of in on at to by with from for as into onto upon within among between under over
and or but if than
whatsoever single exception creature creatures object objects person persons people
""".split())

_NEG = re.compile(r"\b(?:not|never|cannot|can't|isn't|aren't|don't|doesn't|won't|wouldn't|fail|fails|lack|lacks)\b|n't\b")

# úvodné obaly vety, ktoré by inak zmiatli detekciu kvantifikátora ("It is certain that no ...")
_WRAP = re.compile(r"""^(?:
    (?:it|this|that)\s+(?:is|was|has\s+been|can\s+be|must\s+be|should\s+be|will\s+be)
        (?:\s+(?:also|clearly|logically|therefore|thus|undeniably|definitely|certainly|always))*
        \s+(?:a\s+)?(?:certain(?:ty)?|true|the\s+case|(?:a\s+)?(?:known\s+|well-known\s+|established\s+|undeniable\s+|clear\s+|logical\s+|simple\s+|proven\s+)?fact|
            known|clear|evident|undeniable|accepted|established|correct|valid|inevitable|necessary|logical|obvious|understood|stated|given|assumed|
            (?:the\s+)?(?:conclusion|inference|result)|concluded|deduced|inferred|implied|apparent)
        (?:\s+(?:fact|truth|statement))?\s+that
  | (?:it\s+)?(?:logically\s+|necessarily\s+|therefore\s+)?(?:follows|follow)(?:\s+(?:from\s+this|logically|necessarily|from\s+the\s+premises|that))*(?:\s+that)?
  | (?:this|that|which|these)\s+(?:means|implies|proves|shows|demonstrates|establishes|indicates|entails|necessitates|leads\s+(?:us\s+)?to(?:\s+(?:the\s+)?conclu\w+)?|results\s+in|allows\s+us\s+to\s+(?:state|conclude|say|infer))(?:\s+that)?
  | (?:one|we|you)\s+can\s+(?:therefore\s+|thus\s+|logically\s+|safely\s+|then\s+)?(?:conclude|infer|deduce|state|say|assert)(?:\s+that)?
  | (?:therefore|thus|hence|consequently|accordingly|so|as\s+a\s+result|in\s+conclusion|finally|ultimately|then)
  | (?:there\s+is\s+(?:a|the)\s+)?(?:logical\s+)?(?:conclusion|inference)\s+that
  | (?:we|you|one)\s+(?:can|may|must)\s+(?:therefore\s+)?(?:see|say|state|conclude)(?:\s+that)?
  )\b[\s,:]*""", re.X)


def clean_sentence(s: str) -> str:
    s = s.strip().rstrip(".!?").lower()
    s = re.sub(r"\s+", " ", s)
    for _ in range(4):                       # obaly môžu byť zreťazené ("Therefore, it is certain that ...")
        n = _WRAP.sub("", s, count=1).strip(" ,")
        if n == s:
            break
        s = n
    return s


# ----------------------------------------------------------------------------------------------- 2. typ výroku
_NOT_ALL = re.compile(r"\bnot\s+(?:all|every|each|everything|everyone|everybody)\b|\bnot\s+(?:a\s+)?(?:all|every|each)\b|"
                      r"\bit\s+is\s+not\s+(?:true|the\s+case)\s+that\s+(?:all|every|each|everything)\b|\bnot\s+every\b")
_E = re.compile(r"\bmutually\s+exclusive\b|\bseparate\s+from\b|\bdistinct\s+from\b|\bno\s+overlap\b|\bdisjoint\b|\bimpossible\b|\bnothing\s+in\s+common\b|\bno\b|\bnone\b|\bnothing\b|\bnot\s+a\s+single\b|\bnot\s+one\b|\bnot\s+any\b|\bnobody\b|\bno\s+one\b|"
                r"\bnever\b|\bit\s+is\s+not\s+true\s+that\s+any\b|\bnot\s+true\s+that\s+any\b|"
                r"\bnot\s+(?:a\s+single\s+)?(?:one|thing)\b|\bnot\s+a\s+single\b|\bis\s+not\s+true\s+that\s+any\b")
_SOME = re.compile(r"\bsome\b|\bcertain\b|\ba\s+portion\b|\ba\s+(?:certain\s+)?number\s+of\b|\bat\s+least\s+(?:some|one|a)\b|"
                   r"\ba\s+subset\b|\ba\s+few\b|\bseveral\b|\bthere\s+(?:exist|exists|is|are)\b|\ba\s+select\b|^a\s+part\s+of\b|"
                   r"\bmany\b|\ba\s+(?:small\s+)?(?:fraction|group|segment|proportion|selection|handful)\b|\bone\s+or\s+more\b|"
                   r"\bsome\s+of\b|\bsome\b|\bcertain\s+\w+\b")
_ALL = re.compile(r"\ball\b|\bevery\b|\beach\b|\bany\b|\banything\b|\beverything\b|\bwhatever\b|\bwhoever\b|\bevery\s+single\b|"
                  r"\bwithout\s+exception\b|\bentire\b|\bwhole\b|\beveryone\b|\beverybody\b|\bsubset\s+of\s+the\s+set\b|"
                  r"\bis\s+a\s+subset\b|\bare\s+a\s+subset\b|\bare\s+subsets\b|\bhas\s+(?:a\s+)?(?:every|all)\b")
_GENERIC_SG = re.compile(r"^(?:an?|the)\s+\w+(?:\s+\w+)?\s+(?:is|are|can\s+be)\b")


_NO_NOT = re.compile(r"\bno\b.*\b(?:that|who|which|whom)\s+(?:is|are|do|does|can|could|would|will|has|have)\s*(?:not|n't)\b|"
                     r"\bno\b.*\b(?:isn't|aren't|doesn't|don't|cannot)\b|\bno\s+such\s+thing\b.*(?:isn't|aren't|not)\b")
_FALSE_ANY = re.compile(r"\bfalse\s+that\s+(?:any|anything)\b|\bfalse\s+that\s+there\s+(?:is|are|exist)\s+any\b")
_FALSE_ALL = re.compile(r"\bfalse\s+that\s+(?:all|every|each|everything)\b")


def statement_type(sent: str) -> tuple[str | None, str]:
    """Vráti (typ, dôvod/pravidlo). Typ ∈ {A,E,I,O} alebo None, ak parser nechce hádať."""
    s = clean_sentence(sent)
    if _NO_NOT.search(s):
        return "A", "double-negation"          # "There are no X that are not Y" == All X are Y
    if _FALSE_ALL.search(s):
        return "O", "false-all"
    if _FALSE_ANY.search(s):
        return "E", "false-any"
    if re.match(r"there\s+(?:are|exist)\s+(?:all|every)\b", s):
        return "A", "there-are-all"
    if _NOT_ALL.search(s):
        return "O", "not-all"
    if _E.search(s):
        return "E", "no"
    if _SOME.search(s):
        return ("O" if _NEG.search(s) else "I"), "some"
    if _ALL.search(s):
        return ("E" if _NEG.search(s) else "A"), "all"      # "all S are not P" -> E (zriedkavé)
    if _GENERIC_SG.search(s):
        return ("E" if _NEG.search(s) else "A"), "generic"  # "A house is a building."
    if re.match(r"^[a-z]+s\s+(?:are|can\s+be)\b", s):
        return ("E" if _NEG.search(s) else "A"), "bare-plural"
    return None, "no-quantifier"


# ----------------------------------------------------------------------------------------------- 3. termíny
@dataclass
class Parse:
    ok: bool
    reason: str = ""
    types: list[str] = field(default_factory=list)      # [premisa1, premisa2, záver]
    # term-id: 'X' (podmet záveru), 'Y' (prísudok záveru), 'Z' (stredný)
    stmts: list[tuple[str, str, str]] = field(default_factory=list)   # (typ, podmet, prísudok)
    spans: dict[str, list[str]] = field(default_factory=dict)         # symbol -> lemmy termínu


_QPHRASE = re.compile(r"\ba\s+(?:certain\s+|small\s+|large\s+)?number\s+of\b|\bthe\s+(?:set|group|category|class|collection)\s+of\b|"
                      r"\ba\s+(?:subset|portion|part|fraction|group|selection)\s+of\b")


def _content(sent: str) -> list[tuple[int, str]]:
    toks = tokenize(_QPHRASE.sub(" ", clean_sentence(sent)))
    return [(i, lemma(t)) for i, t in enumerate(toks) if t.isalpha() and t not in _STOP and lemma(t) not in _STOP]


def parse_syllogism(text: str) -> Parse:
    sents = split_sentences(text)
    if len(sents) != 3:
        return Parse(False, f"sentences={len(sents)}")
    types = []
    for s in sents:
        t, why = statement_type(s)
        if t is None:
            return Parse(False, "type:" + why, types)
        types.append(t)
    c = [_content(s) for s in sents]
    A1, A2, C = ({l for _, l in x} for x in c)
    M = A1 & A2
    if not M:
        return Parse(False, "no-middle-term", types)
    T1, T2 = A1 - M, A2 - M
    if not T1 or not T2:
        return Parse(False, "empty-term", types)
    if T1 & T2:
        return Parse(False, "terms-overlap", types)

    def first_pos(content, toks):
        pos = [i for i, l in content if l in toks]
        return min(pos) if pos else None

    # podmet/prísudok v každej premise: termín, ktorý sa vyskytuje skôr
    def order(content, a, b):
        pa, pb = first_pos(content, a), first_pos(content, b)
        return (a, b) if pa <= pb else (b, a)

    pc1, pc2, pcc = c
    s1, p1 = order(pc1, T1, M)
    s2, p2 = order(pc2, T2, M)
    h1, h2 = bool(C & T1), bool(C & T2)
    if not (h1 and h2):
        return Parse(False, "conclusion-terms-not-in-premises", types)
    if C & M and not (C & T1 and C & T2):
        return Parse(False, "conclusion-mentions-middle", types)
    cs, cp = order(pcc, T1, T2)
    name = {id(T1): None}
    sym = {}
    sym_for = lambda tokset: "X" if tokset is cs else ("Y" if tokset is cp else "Z")
    stm = [(types[0], sym_for(s1), sym_for(p1)), (types[1], sym_for(s2), sym_for(p2)), (types[2], sym_for(cs), sym_for(cp))]
    return Parse(True, "", types, stm, {"X": sorted(cs), "Y": sorted(cp), "Z": sorted(M)})


# ----------------------------------------------------------------------------------------------- 4. abstrakcia
_JOIN = {"of", "in", "on", "at", "to", "by", "with", "from", "for", "as", "the", "a", "an"}


def abstract_text(text: str, parse: Parse) -> str:
    """Nahradí obsahové termíny symbolmi X/Y/Z v pôvodnom znení viet (dvojpriechodovo).
    1. priechod: každému tokenu priradí symbol (podľa lemmy termínu) alebo None;
    2. priechod: úseky tokenov s rovnakým symbolom, oddelené iba predložkami/členmi, zlúči do jedného symbolu."""
    owner = {l: sym for sym, ls in parse.spans.items() for l in ls}
    out = []
    for sent in split_sentences(text):
        toks = [t for t in re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+|[^\w\s]|\s+", sent) if not t.isspace()]
        syms = [owner.get(lemma(t.lower())) if t.isalpha() else None for t in toks]
        res, i = [], 0
        while i < len(toks):
            if syms[i]:
                s0, j = syms[i], i
                k = i + 1
                while k < len(toks):                          # pokus o rozšírenie úseku
                    m = k
                    while m < len(toks) and toks[m].lower() in _JOIN and syms[m] is None:
                        m += 1
                    if m < len(toks) and syms[m] == s0:
                        j = k = m + 1
                    else:
                        break
                res.append(s0); i = j + 1 if j == i else j
            else:
                res.append(toks[i]); i += 1
        txt = " ".join(res)
        txt = re.sub(r"\s+([,.;:!?])", r"\1", txt)
        out.append(txt)
    return " ".join(out)


_CANON = {"A": "All {x} are {y}.", "E": "No {x} are {y}.", "I": "Some {x} are {y}.", "O": "Some {x} are not {y}."}


def canonical_text(parse: Parse) -> str:
    (t1, a1, b1), (t2, a2, b2), (t3, a3, b3) = parse.stmts
    # hlavná premisa (obsahuje prísudok záveru Y) ide prvá
    prem = sorted(parse.stmts[:2], key=lambda s: 0 if "Y" in (s[1], s[2]) else 1)
    f = lambda s: _CANON[s[0]].format(x=s[1], y=s[2])
    return f"Premise 1: {f(prem[0])}\nPremise 2: {f(prem[1])}\nConclusion: {f(parse.stmts[2])}"


# ----------------------------------------------------------------------------------------------- 5. platnosť
def _holds(kind: str, x: int, y: int, occupied: list[int]) -> bool:
    """Pravdivosť výroku v modeli: `occupied` = zoznam neprázdnych Vennových oblastí (3-bitové masky)."""
    if kind == "A": return all(((r >> y) & 1) for r in occupied if (r >> x) & 1)
    if kind == "E": return not any(((r >> x) & 1) and ((r >> y) & 1) for r in occupied)
    if kind == "I": return any(((r >> x) & 1) and ((r >> y) & 1) for r in occupied)
    if kind == "O": return any(((r >> x) & 1) and not ((r >> y) & 1) for r in occupied)
    raise ValueError(kind)


_IDX = {"X": 0, "Y": 1, "Z": 2}
_MODELS = [[r for r in range(8) if (m >> r) & 1] for m in range(256)]
_MODELS = [m for m in _MODELS if all(any((r >> t) & 1 for r in m) for t in range(3))]   # existenčný predpoklad


def is_valid_stmts(stmts, existential_import: bool = True) -> bool:
    models = _MODELS if existential_import else [[r for r in range(8) if (m >> r) & 1] for m in range(256)]
    (k1, a1, b1), (k2, a2, b2), (k3, a3, b3) = stmts
    for m in models:
        if _holds(k1, _IDX[a1], _IDX[b1], m) and _holds(k2, _IDX[a2], _IDX[b2], m) and not _holds(k3, _IDX[a3], _IDX[b3], m):
            return False
    return True


def count_valid_forms() -> int:
    """Kontrola správnosti: pri existenčnom predpoklade musí byť platných presne 24 zo 256 mód-figúra kombinácií."""
    n = 0
    for k1, k2, k3, fig in product("AEIO", "AEIO", "AEIO", range(4)):
        # figúra určuje polohu stredného termínu Z; záver je vždy X-Y
        prem = [(("Z", "Y"), ("X", "Z")), (("Y", "Z"), ("X", "Z")), (("Z", "Y"), ("Z", "X")), (("Y", "Z"), ("Z", "X"))][fig]
        if is_valid_stmts([(k1, *prem[0]), (k2, *prem[1]), (k3, "X", "Y")]):
            n += 1
    return n
