"""Shared provenance and evaluation for neural experiments (no model imports)."""
import hashlib
import importlib.metadata
import json
import platform
from .evaluate import official_scores, bootstrap_ci, confusion
from .utils import save_json


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment():
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'sentencepiece', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def resolve_revision(model, revision):
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision).sha


def evaluate_predictions(gold, predictions, cfg):
    ids = [x['id'] for x in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {x['id'] for x in gold}:
        raise ValueError('Predictions must cover all gold IDs exactly once.')
    if any(p['validity'] is not None and type(p['validity']) is not bool for p in predictions):
        raise ValueError('Prediction must be bool or None (abstention).')
    missing = sum(p['validity'] is None for p in predictions)
    if missing:
        pm = {p['id']: p['validity'] for p in predictions}
        return {'n': len(gold), 'unparsable_answers': missing,
                'answer_coverage': (len(gold) - missing) / len(gold),
                'accuracy_with_abstentions_as_errors': 100 * sum(pm[g['id']] == g['validity'] for g in gold) / len(gold),
                'official_scores': None,
                'note': 'Official metrics withheld: missing binary answers; raw predictions retain null.'}
    return {'n': len(gold), 'unparsable_answers': 0, 'answer_coverage': 1.0,
            'official_scores': official_scores(gold, predictions),
            'ci95': bootstrap_ci(gold, predictions, cfg['bootstrap']['n_resamples'], cfg['bootstrap']['alpha'], cfg['seed']),
            'confusion': confusion(gold, predictions)}


def save_evaluation(path, gold, predictions, cfg, metadata):
    result = {**metadata, 'data_sha256': fingerprint(gold), **evaluate_predictions(gold, predictions, cfg)}
    save_json(result, path)
    return result
