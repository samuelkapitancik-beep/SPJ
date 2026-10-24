"""Predspracovanie: delenie na vety a základná normalizácia (bez lematizácie - kvantifikátory nesmú zmiznúť)."""
import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def n_premises(text: str) -> int:
    """Posledná veta je záver; ostatné sú premisy (platí pre ST2 formát)."""
    return max(len(split_sentences(text)) - 1, 0)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())
