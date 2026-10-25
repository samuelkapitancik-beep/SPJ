"""Načítanie konfigurácie ST2: spoločný configs/experiment.yaml (NEMENÍ sa) + configs/st2.yaml (sekcia `st2`)."""
from __future__ import annotations
from .utils import load_config

ST2_CONFIG = "configs/st2.yaml"


def load_st2_config(base: str = "configs/experiment.yaml", st2: str = ST2_CONFIG) -> dict:
    cfg = load_config(base)
    extra = load_config(st2)
    if "st2" in cfg:
        raise ValueError(f"{base} už obsahuje sekciu st2 - ST2 nastavenia patria iba do {st2}")
    cfg["st2"] = extra["st2"]
    return cfg
