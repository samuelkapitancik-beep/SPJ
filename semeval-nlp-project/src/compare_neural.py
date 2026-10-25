"""Build a comparison table from measured results only; separate dev and test."""
import argparse
import statistics
from pathlib import Path
from .utils import ROOT, save_json
from .data_loader import load_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neural-dir', default='outputs/neural')
    a = p.parse_args()
    directory = ROOT / a.neural_dir
    rows = []
    base = load_json('outputs/baseline_results.json')
    for split, key in [('dev', 'dev'), ('test_st1', 'test_st1')]:
        for row in base[key]:
            rows.append({'model': row['model'], 'split': split,
                         **{k: row[k] for k in ('accuracy', 'tce', 'combined_score')}})
    symbolic = load_json('outputs/parser_eval.json')
    for split in ('dev', 'test_st1'):
        rows.append({'model': 'S0', 'split': split, **symbolic[split]['S0_reference_all_examples']})
    for path in sorted(directory.rglob('*metrics.json')):
        result = load_json(path)
        scores = result['official_scores']
        row = {'model': result['model'], 'split': result['split'],
               'seed': result.get('seed'), 'representation': result.get('representation'),
               'source': str(path), 'unparsable_answers': result['unparsable_answers']}
        if scores:
            row.update({k: scores[k] for k in ('accuracy', 'tce', 'combined_score')})
        else:
            row.update(accuracy=result['accuracy_with_abstentions_as_errors'], tce=None, combined_score=None)
        rows.append(row)
    aggregates = []
    for split in ('dev', 'test_st1'):
        seeds = [x for x in rows if x['split'] == split and x.get('seed') is not None
                 and x.get('representation') is None and x.get('combined_score') is not None]
        for model in sorted({x['model'] for x in seeds}):
            group = [x for x in seeds if x['model'] == model]
            aggregates.append({'model': model, 'split': split, 'n_seeds': len(group),
                **{k: {'mean': statistics.mean(x[k] for x in group),
                       'sample_sd': statistics.stdev(x[k] for x in group) if len(group) > 1 else None}
                   for k in ('accuracy', 'tce', 'combined_score')}})
    save_json({'rows': rows, 'across_seeds': aggregates}, directory / 'comparison.json')
    lines = ['# Namerané výsledky', '', '| Model | Split | Seed / vstup | ACC | TCE | CS |',
             '|---|---|---|---:|---:|---:|']
    fmt = lambda value: '—' if value is None else f'{value:.2f}'
    for x in rows:
        label = x.get('representation') or (str(x['seed']) if x.get('seed') is not None else '—')
        lines.append(f'| {x["model"]} | {x["split"]} | {label} | {fmt(x["accuracy"])} | {fmt(x["tce"])} | {fmt(x["combined_score"])} |')
    lines += ['', 'Chýbajúce metriky znamenajú nejednoznačné odpovede LLM; v ACC sú počítané ako chyby.',
              'Priemery a smerodajné odchýlky medzi seedmi sú v comparison.json.']
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
