"""Real backward regression; runs in the neural/Colab environment without downloading weights."""
import os
os.environ['USE_TF'] = '0'
import pytest


def test_deberta_relative_attention_checkpoint_gradients():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from src.finetune_encoder import enable_checkpointing
    config = transformers.DebertaV2Config(
        vocab_size=64, hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=32, relative_attention=True,
        pos_att_type=['p2c', 'c2p'], norm_rel_ebd='layer_norm',
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        pooler_dropout=0.0, cls_dropout=0.0, num_labels=2)
    torch.manual_seed(42)
    plain = transformers.DebertaV2ForSequenceClassification(config).train()
    checkpointed = transformers.DebertaV2ForSequenceClassification(config).train()
    checkpointed.load_state_dict(plain.state_dict())
    enable_checkpointing(checkpointed)
    # Accumulate multiple microbatches, including the shared relative-position path.
    for _ in range(2):
        batch = {'input_ids': torch.randint(1, 64, (2, 12)),
                 'attention_mask': torch.ones(2, 12, dtype=torch.long),
                 'labels': torch.tensor([0, 1])}
        (plain(**batch).loss / 2).backward()
        (checkpointed(**batch).loss / 2).backward()
    for (name, expected), (_, actual) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            assert actual.grad is not None and torch.isfinite(actual.grad).all(), name
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6, msg=name)
