"""Audit uploaded predictions against the frozen local dataset; never train models."""
import json, zipfile, hashlib, math, statistics, argparse
from pathlib import Path
from src.utils import ROOT, load_config, save_json
from src.data_loader import load_json
from src.neural_common import fingerprint, evaluate_predictions
from src.neurosymbolic import build_item, extract_label, PROMPT_SHA
from src.evaluate import official_scores, mcnemar_exact

ARCHIVE = ROOT / 'reproduction/semeval_neural_results.zip'
OUT = ROOT / 'reports/neural_results_review'


def main():
    global ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    ARCHIVE = parser.parse_args().archive
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    data = {'dev': load_json('data/processed/dev.json'), 'test_st1': load_json(cfg['paths']['test_st1'])}
    train = load_json('data/processed/train.json')
    checks, issues, runs, predictions = [], [], [], {}
    def check(name, ok):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok: issues.append(name)
    def compare(name, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                check(name + '.' + k + '.exists', k in b)
                if k in b: compare(name + '.' + k, a[k], b[k])
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            check(name + '.length', len(a) == len(b))
            for i, (x, y) in enumerate(zip(a, b)): compare(f'{name}.{i}', x, y)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            check(name, math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8))
        else: check(name, a == b)
    with zipfile.ZipFile(ARCHIVE) as z:
        read = lambda name: json.loads(z.read(name))
        for seed in range(5):
            prefix = f'b2/seed_{seed}'
            meta, history = read(prefix + '/training.json'), read(prefix + '/history.json')
            check(prefix + ' train hash', meta['train_sha256'] == fingerprint(train))
            check(prefix + ' dev hash', meta['dev_sha256'] == fingerprint(data['dev']))
            check(prefix + ' five epochs', [x['epoch'] for x in history] == [1, 2, 3, 4, 5])
            best = max(history, key=lambda x: x['accuracy'])
            check(prefix + ' earliest best dev epoch', meta['best_epoch'] == best['epoch'])
            for split, key in [('dev', 'dev'), ('test_st1', 'test')]:
                pred, metrics = read(prefix + f'/{key}_predictions.json'), read(prefix + f'/{key}_metrics.json')
                compare(prefix + '/' + split + ' metrics', evaluate_predictions(data[split], pred, cfg), metrics)
                check(prefix + '/' + split + ' data hash', metrics['data_sha256'] == fingerprint(data[split]))
                if split == 'dev': compare(prefix + ' selected dev ACC', best['accuracy'], metrics['official_scores']['accuracy'])
                if split == 'test_st1': compare(prefix + ' test training metadata', meta, metrics['training'])
                run = {'method': 'B2', 'seed': seed, 'split': split, 'best_epoch': best['epoch'], 'metadata': meta,
                       **evaluate_predictions(data[split], pred, cfg)}
                runs.append(run); predictions[f'B2_{seed}/{split}'] = pred
        for split in data:
            items = [build_item(x) for x in data[split]]
            for rep in ['original', 'abstract', 'canonical']:
                prefix = f'llama/{split}/{rep}'
                meta, pred, metrics = read(prefix + '/run.json'), read(prefix + '/predictions.json'), read(prefix + '/metrics.json')
                journal = [json.loads(s) for s in z.read(prefix + '/responses.jsonl').decode().splitlines()]
                check(prefix + ' journal equals predictions', journal == pred)
                check(prefix + ' submission equals predictions', read(prefix + '/submission.json') == [{'id': x['id'], 'validity': x['validity']} for x in pred])
                check(prefix + ' labels match raw responses', all(extract_label(x['raw']) == x['validity'] for x in pred))
                check(prefix + ' input hash', meta['items_sha256'] == fingerprint(items))
                check(prefix + ' data hash', metrics['data_sha256'] == fingerprint(data[split]))
                check(prefix + ' prompt hash', meta['prompt_sha256_12'] == PROMPT_SHA)
                check(prefix + ' fallback representation', all(p['used_representation'] == (rep if it['parsed'] else 'original') for it, p in zip(items, pred)))
                compare(prefix + ' metrics', evaluate_predictions(data[split], pred, cfg), metrics)
                compare(prefix + ' parser coverage', sum(x['parsed'] for x in items) / len(items), metrics['parser_coverage'])
                run = {'method': 'Llama', 'representation': rep, 'split': split, 'metadata': meta,
                       'predicted_valid': sum(p['validity'] for p in pred), **evaluate_predictions(data[split], pred, cfg)}
                for status in [True, False]:
                    subset = [(g,p) for g,p,it in zip(data[split],pred,items) if it['parsed'] == status]
                    run['parsed' if status else 'fallback'] = {'n': len(subset), 'accuracy': 100 * sum(g['validity']==p['validity'] for g,p in subset)/len(subset)}
                runs.append(run); predictions[f'Llama_{rep}/{split}'] = pred
            original = {x['id']: x for x in predictions[f'Llama_original/{split}']}
            for rep in ['abstract','canonical']:
                check(f'{split}/{rep} identical fallback answers', all(p['raw'] == original[p['id']]['raw'] for it,p in zip(items,predictions[f'Llama_{rep}/{split}']) if not it['parsed']))
        supplied = read('comparison.json')
        for r in supplied['rows']:
            if r['model'] == 'microsoft/deberta-v3-base': key = f'B2_{r["seed"]}/{r["split"]}'
            elif r['model'] == 'meta-llama/Meta-Llama-3-8B-Instruct': key = f'Llama_{r["representation"]}/{r["split"]}'
            else: continue
            score = official_scores(data[r['split']], predictions[key])
            for k in ['accuracy','tce','combined_score']: compare('comparison.'+key+'.'+k, score[k], r[k])
    aggregates = {}
    for split in data:
        group = [r for r in runs if r['method']=='B2' and r['split']==split]
        aggregates[split] = {k: {'mean': statistics.mean(r['official_scores'][k] for r in group),
                               'sd': statistics.stdev(r['official_scores'][k] for r in group),
                               'min': min(r['official_scores'][k] for r in group),
                               'max': max(r['official_scores'][k] for r in group)} for k in ['accuracy','tce','combined_score']}
    pairs = []
    for a,b in [('Llama_canonical','Llama_original'),('Llama_abstract','Llama_original'),('B2_0','B1')]:
        pa = predictions[a+'/test_st1']
        pb = load_json('outputs/predictions/st1_B1_tfidf_lr.json') if b == 'B1' else predictions[b+'/test_st1']
        pairs.append({'a': a, 'b': b, **mcnemar_exact(data['test_st1'],pa,pb)})
    prev = 0
    for i, row in enumerate(sorted(pairs,key=lambda x:x['p_value'])):
        prev = max(prev, min(1, (len(pairs)-i)*row['p_value']))
        row['p_holm'] = prev
    result = {'archive':str(ARCHIVE), 'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              'checks':checks,'issues':issues,'runs':runs,'b2_summary':aggregates,'exploratory_mcnemar':pairs,
              'limitations':['No model weights, full console logs or training source snapshot in archive.',
                             'Recomputed scores verify predictions, not independent re-execution of GPU inference.',
                             'Cannot prove absence of test-driven tuning from exported files.']}
    save_json(result, OUT/'audit.json')
    print(json.dumps({'checks':len(checks),'issues':issues,'B2':aggregates,'pairs':pairs,
        'llama':[{'split':r['split'],'rep':r['representation'],'valid':r['predicted_valid'], 'scores':r['official_scores'], 'parsed':r['parsed'],'fallback':r['fallback']} for r in runs if r['method']=='Llama']},indent=2))


if __name__ == '__main__': main()
