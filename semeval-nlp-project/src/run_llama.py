"""Llama zero-shot ablation: original / abstract / canonical, fixed prompt v1."""
import argparse
import json
from .utils import ROOT, load_config, save_json, set_seed
from .data_loader import load_json
from .neurosymbolic import build_item, make_messages, extract_label, PROMPT_SHA, PROMPT_V1
from .neural_common import environment, fingerprint, resolve_revision, save_evaluation


class LlamaRunner:
    def __init__(self, model_id, revision, quantize=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('Llama experiments require a Colab GPU runtime.')
        self.torch = torch
        self.revision = resolve_revision(model_id, revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {'revision': self.revision, 'torch_dtype': dtype, 'device_map': 'auto'}
        if quantize:
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    def predict(self, item, representation, max_new_tokens, fallback='original'):
        tok = self.tokenizer
        inputs = tok.apply_chat_template(make_messages(item, representation, fallback),
                    add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors='pt').to(self.model.device)
        # Refuse truncation: dropping premises would change the logic task.
        if inputs['input_ids'].shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError(f'Argument exceeds context window: {item["id"]}')
        eos = [tok.eos_token_id]
        eot = tok.convert_tokens_to_ids('<|eot_id|>')
        if eot is not None and eot != tok.unk_token_id and eot not in eos:
            eos.append(eot)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                         eos_token_id=eos, pad_token_id=tok.eos_token_id)
        raw = tok.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        label = extract_label(raw)
        return {'id': item['id'], 'validity': label, 'raw': raw,
                'unparsable_answer': label is None, 'parsed': item['parsed'],
                'used_representation': representation if item['parsed'] else 'original'}


def cached_predictions(path, items):
    if not path.exists(): return []
    rows = []
    lines = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1: raise
            # A killed runtime can leave one incomplete trailing write.
            path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
    if [x['id'] for x in rows] != [x['id'] for x in items[:len(rows)]] or len(rows) > len(items):
        raise ValueError('Cached predictions do not match the input order.')
    if rows and not path.read_text(encoding='utf-8').endswith('\n'):
        with path.open('a', encoding='utf-8') as stream: stream.write('\n')
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    p.add_argument('--revision', default='main')
    p.add_argument('--split', choices=['dev', 'test_st1'], default='dev')
    p.add_argument('--representations', nargs='+', choices=['original', 'abstract', 'canonical'],
                   default=['original', 'abstract', 'canonical'])
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--no-4bit', action='store_true')
    p.add_argument('--output', default='outputs/neural/llama')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.max_new_tokens < 1: raise ValueError('max-new-tokens must be positive.')
    cfg = load_config()
    set_seed(cfg['seed'])
    gold = load_json('data/processed/dev.json' if a.split == 'dev' else cfg['paths']['test_st1'])
    items = [build_item(x) for x in gold]
    runner = LlamaRunner(a.model, a.revision, quantize=not a.no_4bit)
    for rep in a.representations:
        dest = ROOT / a.output / a.split / rep
        dest.mkdir(parents=True, exist_ok=True)
        identity = {'experiment': 'N1', 'model': a.model, 'revision': runner.revision,
                    'split': a.split, 'representation': rep, 'prompt_sha256_12': PROMPT_SHA,
                    'prompt': PROMPT_V1, 'items_sha256': fingerprint(items),
                    'quantization': 'nf4-double' if not a.no_4bit else 'none',
                    'max_new_tokens': a.max_new_tokens, 'do_sample': False, 'seed': cfg['seed'],
                    'environment': environment()}
        identity_path = dest / 'run.json'
        if identity_path.exists() and load_json(identity_path) != identity:
            raise ValueError(f'Run configuration changed. Use a new --output directory: {dest}')
        save_json(identity, identity_path)
        journal = dest / 'responses.jsonl'
        preds = cached_predictions(journal, items)
        with journal.open('a', encoding='utf-8') as stream:
            for item in items[len(preds):]:
                result = runner.predict(item, rep, a.max_new_tokens)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                preds.append(result)
                if len(preds) % 20 == 0: print(f'{a.split}/{rep}: {len(preds)}/{len(items)}', flush=True)
        save_json(preds, dest / 'predictions.json')
        if all(p['validity'] is not None for p in preds):
            save_json([{'id': p['id'], 'validity': p['validity']} for p in preds], dest / 'submission.json')
        save_evaluation(dest / 'metrics.json', gold, preds, cfg,
                        {**identity, 'parser_coverage': sum(x['parsed'] for x in items) / len(items)})


if __name__ == '__main__':
    main()
