"""ST2: pomocné textové funkcie nad parserom zo ST1 (src/symbolic.py sa NEMENÍ, iba sa z neho importuje).

  * `content_lemmas`   - množina obsahových lemm vety (presne pravidlá parsera `_content`),
  * `abstract_lexical` - lexikálna abstrakcia celého ST2 textu pre N1 (obsahové lemmy -> umelé slová).
"""
from __future__ import annotations
import random
import re
from .preprocessing import split_sentences
from .symbolic import (_content, _STOP, lemma, _WRAP, _NOT_ALL, _E, _SOME, _ALL, _GENERIC_SG, _NO_NOT,
                       _FALSE_ANY, _FALSE_ALL, _NEG, _QPHRASE)


def content_lemmas(sent: str) -> set[str]:
    """Množina obsahových lemm vety (rovnaké pravidlá ako pri hľadaní termínov v parseri)."""
    return {l for _, l in _content(sent)}


def _pseudowords() -> list[str]:
    """Deterministický zoznam umelých slov CVCVC (napr. 'bakin'). Nekončia na -s/-e (lemma() by ich skrátila),
    nie sú v _STOP, takže parser ich vidí ako obsahové slová s nezmenenou lemmou (aj s pridaným -s)."""
    C, V = "bdfgklmnprtvz", "aiou"
    words = [a + b + c + d + e for a in C for b in V for c in C for d in V for e in "dgklmnprtz"]
    random.Random(11).shuffle(words)
    return [w for w in words if w not in _STOP and lemma(w) == w and lemma(w + "s") == w]


PSEUDOWORDS = _pseudowords()

# slová, ktoré používajú regexy typu výroku a obalov ("without exception", "entire", ...): abstrakcia ich nesmie prepísať,
# inak by sa zmenil typ výroku (A/E/I/O) určený parserom. Porovnáva sa podľa lemmy, aby sa tvary ("number"/"numbers")
# správali rovnako vo všetkých vetách (inak by sa rozbil stredný termín).
_PROTECTED = {lemma(w) for pat in (_WRAP, _NOT_ALL, _E, _SOME, _ALL, _GENERIC_SG, _NO_NOT, _FALSE_ANY, _FALSE_ALL, _NEG, _QPHRASE)
              for w in re.findall(r"[a-z]{3,}", re.sub(r"\\[a-zA-Z]", " ", pat.pattern))}


def abstract_lexical(text: str) -> str:
    """Lexikálna abstrakcia celého textu (ST2): každá obsahová lemma -> umelé slovo, konzistentne naprieč vetami.

    Na rozdiel od `symbolic.abstract_text` (X/Y/Z) nepotrebuje úspešný parse - je teda použiteľná aj pre príklady,
    kde parser nenašiel sylogizmus, a pri 5–7 premisách s rušivými vetami je termínov viac než tri. Nahrádzajú sa
    presne lemmy, ktoré `_content` považuje za obsahové, takže štruktúra, ktorú vidí parser, ostáva zachovaná
    (overené testom na ST1 train+dev). Slová sa priraďujú podľa prvého výskytu."""
    sym: dict[str, str] = {}
    out = []
    for sent in split_sentences(text):
        keep = content_lemmas(sent)
        res = []
        for t in re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+|[^\w\s]|\s+", sent):
            low = t.lower()
            if t.isalpha() and low not in _STOP and lemma(low) not in _PROTECTED and lemma(low) in keep:
                l = lemma(low)
                if l not in sym:
                    sym[l] = PSEUDOWORDS[len(sym)]
                # množné číslo sa zachová koncovkou -s (parser podľa nej rozpoznáva napr. "Books are not ...")
                res.append(sym[l] + ("s" if low.endswith("s") and l != low else ""))
            else:
                res.append(t)
        out.append("".join(res))
    return " ".join(out)
