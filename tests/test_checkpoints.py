"""Validate pretrained backbone coverage without weakening checkpoint checks."""
import unittest

import torch
from models import DiT
from rebuttal.checkpoints import GAMMA, MODE_EMBEDDING, load_compatible


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = DiT(input_size=4, hidden_size=32, depth=2, num_heads=4)
        self.state = {key: value.clone() for key, value in self.model.state_dict().items()
                      if not GAMMA.fullmatch(key) and key != MODE_EMBEDDING}

    def test_accepts_exact_identity_adapters_and_unused_embedding(self):
        report = load_compatible(self.model, {'ema': self.state})
        self.assertEqual(len(report['missing_initialized_identity_or_unused']), 9)
        for key, value in self.model.state_dict().items():
            if GAMMA.fullmatch(key):
                self.assertTrue(bool(torch.all(value == 1)))
            elif key in self.state:
                self.assertTrue(torch.equal(value, self.state[key]))

    def test_missing_backbone_is_rejected(self):
        del self.state['blocks.0.attn.qkv.weight']
        with self.assertRaisesRegex(ValueError, 'missing_required'):
            load_compatible(self.model, self.state)

    def test_unknown_gamma_like_key_is_rejected(self):
        self.state['blocks.0.attn.gamma_unsafe'] = torch.ones(32)
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            load_compatible(self.model, self.state)

    def test_wrong_shape_is_rejected(self):
        self.state['blocks.0.attn.qkv.weight'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'shape_mismatch'):
            load_compatible(self.model, self.state)

    def test_missing_nonidentity_adapter_is_rejected(self):
        with torch.no_grad():
            self.model.blocks[0].gamma1.fill_(2)
        with self.assertRaisesRegex(ValueError, 'initialized to identity'):
            load_compatible(self.model, self.state)

    def test_used_mode_embedding_cannot_be_missing(self):
        with self.assertRaises(ValueError):
            load_compatible(self.model, self.state, mode_embedding_unused=False)

    def test_checkpoint_supplied_adapters_are_preserved(self):
        self.state['blocks.0.gamma1'] = torch.full((32,), 2.)
        load_compatible(self.model, {'model': self.state})
        self.assertTrue(bool(torch.all(self.model.blocks[0].gamma1 == 2)))

    def test_identity_adapters_and_unused_embedding_preserve_forward(self):
        # Nonzero final/modulation weights make this an actual computation check.
        self.state = {key: torch.randn_like(value) * .03 for key, value in self.state.items()}
        load_compatible(self.model, self.state)
        self.model.eval()
        x, t, y = torch.randn(2, 4, 4, 4), torch.tensor([5, 9]), torch.tensor([1, 2])
        with torch.no_grad():
            actual = self.model(x, t, y)
            self.assertGreater(float(actual.abs().sum()), 0)
            for block in self.model.blocks:
                for module in [block, block.attn]:
                    for name in ['gamma1', 'gamma2']:
                        delattr(module, name)
                        setattr(module, name, 1)
            self.model.m_embedder.embedding_table.weight.fill_(99)
            reference = self.model(x, t, y)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
