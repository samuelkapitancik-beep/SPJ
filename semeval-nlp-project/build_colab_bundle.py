"""Build an uploadable Colab notebook and ZIP without environments or model weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})
    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)})
    md('''# SemEval Task 11 — B2 DeBERTa a Llama 3 8B
V Colabe zvoľte **Runtime → Change runtime type → GPU** (T4 alebo výkonnejšiu).
Nahrajte priložený `semeval_colab.zip`. Notebook trénuje B2 na 5 seedoch a porovnáva
tri vstupy Llamy. Predchádzajúce behy boli vyhodnotené v README; tento notebook pripravuje nový beh.
Llama potrebuje schválený prístup k modelu na Hugging Face. Token vložte do Colab Secrets
ako `HF_TOKEN`; nevkladajte ho do kódu. GPU pamäť sa šetrí 4-bitovou NF4 kvantizáciou.
Výstupy a checkpointy sa ukladajú na váš Google Drive. Tréning môže trvať hodiny.
Spúšťajte bunky postupne. Testovacia časť je oddelená a predvolene vypnutá.
''')
    code('''from google.colab import files, drive
from pathlib import Path
import os, zipfile, subprocess, sys, json
uploaded = files.upload()  # vyberte semeval_colab.zip
archive = next(name for name in uploaded if name.endswith('.zip'))
root = Path('/content/semeval-nlp-project')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for entry in z.infolist():
        target = (root / entry.filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Unsafe archive path')
    z.extractall(root)
os.chdir(root)
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/SPJ_SemEval2026/neural_v2')
OUT.mkdir(parents=True, exist_ok=True)
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)
print('Projekt:', root, '\\nVýstupy:', OUT)
''')
    md('## Závislosti a kontrola GPU\nColab už poskytuje GPU PyTorch. Po inštalácii pri výzve reštartujte session a znovu spustite prvú bunku.')
    code('''run('-m', 'pip', 'install', '-r', 'requirements.txt', '-r', 'requirements-neural.txt')
import torch
assert torch.cuda.is_available(), 'Zapnite GPU runtime.'
print(torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
(OUT / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
run('-m', 'pytest', '-q', 'tests')
''')
    md('## Prístup a zmrazenie verzií\nPrístup k [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) musí byť schválený pre váš HF účet. Táto bunka overí prístup pred tréningom a uloží presné revízie.')
    code('''from google.colab import userdata
from huggingface_hub import HfApi, hf_hub_download
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
models = {'b2': 'microsoft/deberta-v3-base', 'llama': 'meta-llama/Meta-Llama-3-8B-Instruct'}
revision_file = OUT / 'model_revisions.json'
if revision_file.exists():
    revisions = json.loads(revision_file.read_text())
else:
    revisions = {k: HfApi().model_info(v).sha for k, v in models.items()}
    revision_file.write_text(json.dumps(revisions, indent=2))
for key, model in models.items():
    hf_hub_download(model, 'config.json', revision=revisions[key])
print('Prístup overený. Revízie:', revisions)
''')
    md('## B2 — tréning a validácia\n5 epoch, 5 seedov, efektívna dávka 16. Najlepšia epocha sa vyberie podľa dev accuracy (pri zhode skoršia). Hotové seedy sa preskočia. Prerušený seed ponechajte na diagnostiku a pre nový pokus použite nový adresár OUT; obnovenie optimalizátora nie je implementované.')
    code('''for seed in range(5):
    if (OUT / 'b2' / f'seed_{seed}' / 'training.json').exists():
        print('Hotový seed, preskakujem:', seed)
        continue
    run('-m', 'src.finetune_encoder', '--mode', 'train', '--seeds', seed,
        '--revision', revisions['b2'], '--output', OUT / 'b2')
''')
    md('## Llama — validácia troch reprezentácií\nRovnaký model a prompt, greedy dekódovanie, maximálne 8 nových tokenov. Nespracované argumenty idú v pôvodnom znení. Odpovede sa ukladajú priebežne; opätovné spustenie doplní chýbajúce.')
    code('''run('-m', 'src.run_llama', '--split', 'dev', '--revision', revisions['llama'], '--output', OUT / 'llama')
run('-m', 'src.compare_neural', '--neural-dir', OUT)
''')
    md('## Záverečný test zmrazených modelov\nPo dokončení validácie a zmrazení postupu nastavte `RUN_FINAL_TEST=True`. Test nepoužívajte na ladenie. B2 načíta uložené checkpointy bez ďalšieho tréningu. Pôvodný verejný test bol už vyhodnotený v baseline; nejde o nový skrytý test.')
    code('''RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    for seed in range(5):
        if not (OUT / 'b2' / f'seed_{seed}' / 'test_metrics.json').exists():
            run('-m', 'src.finetune_encoder', '--mode', 'test', '--seeds', seed, '--output', OUT / 'b2')
    run('-m', 'src.run_llama', '--split', 'test_st1', '--revision', revisions['llama'], '--output', OUT / 'llama')
    run('-m', 'src.compare_neural', '--neural-dir', OUT)
else:
    print('Test zatiaľ nespustený.')
''')
    md('## Stiahnuť výsledky\nArchív obsahuje metriky, surové odpovede a predikcie. Veľké checkpointy zostávajú na Drive. Nejednoznačné odpovede Llamy sa počítajú ako chyby v diagnostickej ACC; oficiálne metriky sa v takom behu nevydávajú, pretože evaluator by ich potichu vynechal.')
    code('''archive = Path('/content/semeval_neural_results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and 'checkpoint' not in path.parts:
            z.write(path, path.relative_to(OUT))
files.download(str(archive))
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'cells': cells}
    for i, cell in enumerate(cells): cell['id'] = f'cell-{i:02d}'
    dest = ROOT / 'colab'
    dest.mkdir(exist_ok=True)
    (dest / 'SemEval_B2_Llama.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    with zipfile.ZipFile(dest / 'semeval_colab.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        paths = []
        for folder in ('src', 'configs', 'tests', 'data/raw', 'data/processed'):
            paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        paths.extend(ROOT / name for name in ('requirements.txt', 'requirements-neural.txt', 'README.md',
                     'reports/etapa2_navod.md', 'outputs/baseline_results.json', 'outputs/parser_eval.json'))
        manifest = {}
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            z.write(path, relative)
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        z.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    print(dest)


if __name__ == '__main__':
    main()
