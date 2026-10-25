"""ST2 baseliny: k predikcii platnosti z existujúcich baselinov (src/baselines.py, NEMENÍ sa) sa doplnia premisy."""
from __future__ import annotations
from .st2_symbolic import overlap_pair


def st2_with_premises(validity_preds: list[dict], data: list[dict], premises: list[list[int]]) -> list[dict]:
    """ST2: k predikcii platnosti doplní premisy (iba pri 'platné'; neplatné majú [] ako v oficiálnych dátach)."""
    vm = {p["id"]: p["validity"] for p in validity_preds}
    return [{"id": x["id"], "validity": bool(vm[x["id"]]), "relevant_premises": (list(r) if vm[x["id"]] else [])}
            for x, r in zip(data, premises)]


def overlap_premises(data: list[dict]) -> list[list[int]]:
    """Referencia pre výber premís (ST2): dvojica s najväčším lexikálnym prekryvom so záverom (bez parsera)."""
    return [overlap_pair(x["syllogism"]) for x in data]
