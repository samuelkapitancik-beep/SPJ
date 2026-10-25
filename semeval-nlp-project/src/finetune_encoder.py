"""B2 DeBERTa: train/dev checkpoint selection, then a separate frozen test run."""
import argparse
import gc
import math
import os
# This experiment uses PyTorch only; avoid importing Colab's unrelated TensorFlow backend.
os.environ['USE_TF'] = '0'
from .utils import ROOT, load_config, set_seed, save_json
from .data_loader import load_json, prepare_splits
from .evaluate import official_scores
from .neural_common import environment, fingerprint, resolve_revision, save_evaluation


def enable_checkpointing(model):
    # DeBERTa shares relative-position tensors across layers. Reentrant checkpoint
    # backward can traverse their graph repeatedly after saved tensors were freed.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'test'], default='train')
    p.add_argument('--model', default='microsoft/deberta-v3-base')
    p.add_argument('--revision', default='main')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--output', default='outputs/neural/b2')
    p.add_argument('--allow-cpu', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if min(a.epochs, a.batch, a.accumulation, a.max_length) < 1 or a.lr <= 0:
        raise ValueError('Training parameters must be positive.')
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Seeds must be unique.')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and not a.allow_cpu:
        raise RuntimeError('Enable a GPU runtime in Colab (or explicitly pass --allow-cpu).')
    cfg = load_config()
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)

    def predict(model, tok, data, max_length):
        model.eval()
        preds = []
        with torch.inference_mode():
            for start in range(0, len(data), a.batch):
                rows = data[start:start + a.batch]
                inputs = tok([x['syllogism'] for x in rows], padding=True, truncation=True,
                             max_length=max_length, return_tensors='pt').to(device)
                labels = model(**inputs).logits.argmax(-1).tolist()
                preds.extend({'id': x['id'], 'validity': bool(y)} for x, y in zip(rows, labels))
        return preds

    if a.mode == 'test':
        test = load_json(cfg['paths']['test_st1'])
        for seed in a.seeds:
            dest = out / f'seed_{seed}'
            meta = load_json(dest / 'training.json')
            if (dest / 'test_metrics.json').exists():
                raise FileExistsError(f'Test already evaluated: {dest}')
            tok = AutoTokenizer.from_pretrained(dest / 'checkpoint')
            model = AutoModelForSequenceClassification.from_pretrained(dest / 'checkpoint').to(device)
            pred = predict(model, tok, test, meta['max_length'])
            save_json(pred, dest / 'test_predictions.json')
            save_evaluation(dest / 'test_metrics.json', test, pred, cfg,
                            {'experiment': 'B2', 'split': 'test_st1', 'seed': seed,
                             'model': meta['model'], 'revision': meta['revision'],
                             'training': meta, 'environment': environment()})
            del model
            gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()
        return

    train = load_json('data/processed/train.json')
    dev = load_json('data/processed/dev.json')
    expected_train, expected_dev, _, _, _ = prepare_splits(cfg)
    if train != expected_train or dev != expected_dev:
        raise ValueError('Processed splits differ from the baseline protocol.')
    revision = resolve_revision(a.model, a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision, use_fast=False)
    for seed in a.seeds:
        dest = out / f'seed_{seed}'
        dest.mkdir(parents=True, exist_ok=False)
        set_seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda': torch.cuda.manual_seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained(a.model, revision=revision, num_labels=2).to(device)
        enable_checkpointing(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        batches = math.ceil(len(train) / a.batch)
        steps = a.epochs * math.ceil(batches / a.accumulation)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        use_amp = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        best = float('-inf')
        history = []
        best_pred = None
        for epoch in range(1, a.epochs + 1):
            model.train()
            order = torch.randperm(len(train)).tolist()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for batch_index, start in enumerate(range(0, len(train), a.batch)):
                rows = [train[i] for i in order[start:start + a.batch]]
                inputs = tok([x['syllogism'] for x in rows], padding=True, truncation=True,
                             max_length=a.max_length, return_tensors='pt').to(device)
                labels = torch.tensor([int(x['validity']) for x in rows], device=device)
                group_start = (batch_index // a.accumulation) * a.accumulation * a.batch
                group_size = min(a.accumulation * a.batch, len(train) - group_start)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    loss = model(**inputs, labels=labels).loss
                loss_total += loss.item() * len(rows)
                scaler.scale(loss * len(rows) / group_size).backward()
                if (batch_index + 1) % a.accumulation == 0 or batch_index + 1 == batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale: scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            pred = predict(model, tok, dev, a.max_length)
            scores = official_scores(dev, pred)
            history.append({'epoch': epoch, 'train_loss': loss_total / len(train), **scores})
            print(f'Seed {seed}, epoch {epoch}: dev ACC={scores["accuracy"]:.2f}, TCE={scores["tce"]:.2f}', flush=True)
            if scores['accuracy'] > best:
                best, best_epoch, best_pred = scores['accuracy'], epoch, pred
                model.save_pretrained(dest / 'checkpoint', safe_serialization=True)
                tok.save_pretrained(dest / 'checkpoint')
            save_json(history, dest / 'history.json')
        metadata = {'experiment': 'B2', 'split': 'dev', 'model': a.model, 'revision': revision,
                    'gradient_checkpointing': {'use_reentrant': False},
                    'seed': seed, 'selection': 'dev_accuracy_earliest_tie', 'best_epoch': best_epoch,
                    'max_length': a.max_length, 'arguments': vars(a), 'environment': environment(),
                    'train_sha256': fingerprint(train), 'dev_sha256': fingerprint(dev)}
        save_json(metadata, dest / 'training.json')
        save_json(best_pred, dest / 'dev_predictions.json')
        save_evaluation(dest / 'dev_metrics.json', dev, best_pred, cfg, metadata)
        del model, optimizer, scheduler, scaler
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
