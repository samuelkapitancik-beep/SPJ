"""Reproduce the existing CPU experiments, preserving supplied reference artifacts."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from urllib.request import urlopen
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
COMMIT = "6f71ea846770f9418c7d2845ec6c25e59f885bd7"
BASE = f"https://raw.githubusercontent.com/neuro-symbolic-ai/semeval_2026_task_11/{COMMIT}/"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    os.chdir(ROOT)
    reference = ROOT / "reproduction/reference"
    if not reference.exists():
        reference.mkdir(parents=True)
        shutil.copytree(ROOT / "outputs", reference / "outputs")
        shutil.copytree(ROOT / "data/processed", reference / "processed")
    logs = ROOT / "reproduction/logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": BASE, "commit": COMMIT, "files": {}}
    files = {
        "train_data/subtask 1/train_data.json": "data/raw/train_data/subtask 1/train_data.json",
        "test_data/subtask 1/test_data_subtask_1.json": "data/raw/test_data/subtask 1/test_data_subtask_1.json",
        "test_data/subtask 2/test_data_subtask_2.json": "data/raw/test_data/subtask 2/test_data_subtask_2.json",
        "evaluation_kit/task 1 & 3/evaluation_script.py": "src/vendor/official_eval_task1_3/evaluation_script.py",
        "evaluation_kit/task 2 & 4/evaluation_script.py": "src/vendor/official_eval_task2_4/evaluation_script.py",
    }
    for remote, local in files.items():
        with urlopen(BASE + quote(remote), timeout=60) as response:
            content = response.read()
        path = ROOT / local
        if "src/vendor" in local:
            # Git may convert LF to CRLF on Windows.
            if path.read_bytes().replace(b"\r\n", b"\n") != content.replace(b"\r\n", b"\n"):
                raise RuntimeError(f"Vendored evaluator differs from upstream: {local}")
        else:
            if path.exists() and read(path) != json.loads(content):
                raise RuntimeError(f"Existing dataset differs from pinned source: {local}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest["files"][local] = hashlib.sha256(content).hexdigest()
    (ROOT / "reproduction/data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONHASHSEED": "42"}
    for name, args in [
        ("tests", ["-m", "pytest", "-q", "tests"]),
        ("train", ["-m", "src.train"]),
        ("diagnostics", ["-m", "src.diagnostics"]),
        ("parser", ["-m", "src.run_parser_eval"]),
    ]:
        print(f"Running {name}...", flush=True)
        with (logs / f"{name}.txt").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements-reproduced.txt").write_text(freeze, encoding="utf-8")
    comparisons = {}
    for old in sorted((reference / "outputs").rglob("*")):
        if old.name in {"parser_audit.json", "baseline_results_run1_dev_selection.json"}:
            continue  # Historical/manual artifacts are preserved, not regenerated.
        if old.suffix not in {".json", ".jsonl"}:
            continue
        rel = old.relative_to(reference / "outputs")
        current = ROOT / "outputs" / rel
        if not current.exists():
            continue
        if old.suffix == ".jsonl":
            a, b = ([json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()] for p in (old, current))
        else:
            a, b = read(old), read(current)
        if str(rel) == "baseline_results.json":
            a = {k: v for k, v in a.items() if k not in {"environment", "config"}}
            b = {k: v for k, v in b.items() if k not in {"environment", "config"}}
        comparisons[str(rel)] = a == b
    for old in (reference / "processed").glob("*.json"):
        comparisons[f"processed/{old.name}"] = read(old) == read(ROOT / "data/processed" / old.name)
    (ROOT / "reproduction/comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    print("Finished. See reproduction/logs and outputs.")


if __name__ == "__main__":
    main()
