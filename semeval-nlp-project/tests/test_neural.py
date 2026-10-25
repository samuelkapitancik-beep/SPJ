import json
import pytest
from src.neural_common import evaluate_predictions, fingerprint
from src.neurosymbolic import build_item, make_messages
from src.run_llama import cached_predictions, parser as llama_parser
from src.finetune_encoder import parser as encoder_parser

CFG = {'bootstrap': {'n_resamples': 20, 'alpha': .05}, 'seed': 42}


def gold():
    return [{'id': str(i), 'validity': v, 'plausibility': p}
            for i, (v, p) in enumerate([(True, True), (True, False), (False, True), (False, False)])]


def test_abstention_not_counted_as_correct_invalid_or_dropped():
    data = gold()
    pred = [{'id': x['id'], 'validity': x['validity']} for x in data]
    pred[-1]['validity'] = None
    result = evaluate_predictions(data, pred, CFG)
    assert result['accuracy_with_abstentions_as_errors'] == 75
    assert result['official_scores'] is None
    assert result['answer_coverage'] == .75


def test_complete_answers_use_official_metrics():
    data = gold()
    pred = [{'id': x['id'], 'validity': x['validity']} for x in data]
    result = evaluate_predictions(data, pred, CFG)
    assert result['official_scores']['accuracy'] == 100
    assert result['official_scores']['tce'] == 0


def test_incomplete_or_duplicate_predictions_rejected():
    data = gold()
    with pytest.raises(ValueError): evaluate_predictions(data, data[:-1], CFG)
    with pytest.raises(ValueError): evaluate_predictions(data, data + data[:1], CFG)


def test_llama_fallback_preserves_text_without_gold_labels():
    item = build_item({'id': '1', 'syllogism': 'Dogs bark. Cats meow.', 'validity': True, 'plausibility': False})
    assert not item['parsed']
    for rep in ('original', 'abstract', 'canonical'):
        assert 'Dogs bark. Cats meow.' in make_messages(item, rep)[1]['content']
    assert 'validity' not in item and 'plausibility' not in item


def test_resume_repairs_partial_last_line_and_rejects_wrong_ids(tmp_path):
    path = tmp_path / 'responses.jsonl'
    path.write_text('{"id": "1", "validity": true}\n{"id":', encoding='utf-8')
    assert len(cached_predictions(path, [{'id': '1'}, {'id': '2'}])) == 1
    assert len(path.read_text().splitlines()) == 1
    with pytest.raises(ValueError): cached_predictions(path, [{'id': '2'}])


def test_default_commands_do_not_evaluate_test():
    assert encoder_parser().parse_args([]).mode == 'train'
    assert llama_parser().parse_args([]).split == 'dev'
    assert encoder_parser().parse_args([]).batch * encoder_parser().parse_args([]).accumulation == 16


def test_input_fingerprint_detects_changes():
    assert fingerprint(gold()) == fingerprint(gold())
    assert fingerprint(gold()) != fingerprint(list(reversed(gold())))


def test_llama_cli_exports_all_variants_and_resumes_without_new_inference(tmp_path, monkeypatch):
    import src.run_llama as module
    from src.data_loader import load_json as real_load
    rows = [{**x, 'syllogism': 'All cats are animals. Every kitten is a cat. Therefore, all kittens are animals.'}
            for x in gold()]
    calls = []
    class FakeRunner:
        revision = 'frozen-test-revision'
        def __init__(self, *args, **kwargs): pass
        def predict(self, item, representation, max_new_tokens):
            calls.append((item['id'], representation))
            return {'id': item['id'], 'validity': True, 'raw': 'VALID', 'unparsable_answer': False}
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'load_config', lambda: CFG)
    monkeypatch.setattr(module, 'LlamaRunner', FakeRunner)
    monkeypatch.setattr(module, 'load_json', lambda path: rows if str(path) == 'data/processed/dev.json' else real_load(path))
    module.main([])
    assert len(calls) == 12
    for rep in ('original', 'abstract', 'canonical'):
        dest = tmp_path / 'outputs/neural/llama/dev' / rep
        metrics = real_load(dest / 'metrics.json')
        assert metrics['official_scores']['accuracy'] == 50
        assert len(real_load(dest / 'submission.json')) == 4
    module.main([])
    assert len(calls) == 12  # All 12 responses came from the journal on the second run.


def test_resume_adds_missing_newline(tmp_path):
    path = tmp_path / 'responses.jsonl'
    path.write_text('{"id":"1","validity":true}', encoding='utf-8')
    cached_predictions(path, [{'id': '1'}])
    assert path.read_text().endswith('\n')
