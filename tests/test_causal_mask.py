import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from blocks import MultiHeadAttention
from gpt import GPT
from transformer import Transformer


def reference_attention(self, query, key, value, mask=None):
    """Use PyTorch's causal kernel as an independent reference for decoder attention."""
    batch_size = query.size(0)
    q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    if self.pos_encoding_type == "rotary":
        q, k = self.rope(q), self.rope(k)
    # These model forwards mask only decoder self-attention. Use is_causal rather
    # than copying the model's mask so the reference independently checks causality.
    output = F.scaled_dot_product_attention(q, k, v, is_causal=mask is not None, dropout_p=0.0)
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
    return self.out_linear(output)


class CausalMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.gpt = GPT(tgt_vocab_size=16, embed_dim=8, num_layers=2, n_heads=2, dropout_rate=0.0).eval()
        self.transformer = Transformer(
            embed_dim=8, src_vocab_size=16, tgt_vocab_size=16, seq_len=8, num_layers=2, n_heads=2
        ).eval()

    def logits(self, model, target, source=None):
        # TFS-4 tracks the encoder-decoder's implicit softmax. Capture its actual
        # vocabulary projection so constant probabilities cannot conceal a leak.
        projections = []
        handle = model.decoder.fully_connected.register_forward_hook(
            lambda module, inputs, output: projections.append(output)
        )
        try:
            if source is None:
                output = model(target, model.make_tgt_mask(target))
            else:
                output = model(source, target)
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(output).all())
        return projections[0]

    def test_masks_block_only_future_positions_and_preserve_existing_shapes(self):
        for batch_size in (1, 2):
            for length in (1, 4):
                for model in (self.gpt, self.transformer):
                    with self.subTest(model=type(model).__name__, batch=batch_size, length=length):
                        target = torch.zeros(batch_size, length, dtype=torch.long)
                        mask = model.make_tgt_mask(target)
                        self.assertEqual(mask.dtype, torch.bool)
                        self.assertEqual(mask.device, target.device)
                        expected = torch.tensor([[key > query for key in range(length)] for query in range(length)])
                        if model is self.transformer:
                            expected = expected.expand(batch_size, 1, length, length)
                        torch.testing.assert_close(mask, expected)

    def test_masks_follow_available_accelerator_devices(self):
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        if not devices:
            self.skipTest("No CUDA or MPS device available")
        for device in devices:
            for model in (self.gpt, self.transformer):
                with self.subTest(device=device, model=type(model).__name__):
                    target = torch.ones(2, 3, dtype=torch.long, device=device)
                    mask = model.make_tgt_mask(target)
                    self.assertEqual(mask.device, target.device)
                    torch.testing.assert_close(mask.cpu(), model.make_tgt_mask(target.cpu()))

    def test_attention_rejects_float_masks_with_a_clear_convention(self):
        attention = MultiHeadAttention(8, 2, "sinusoidal", dropout_rate=0.0)
        inputs = torch.randn(2, 3, 8)
        with self.assertRaisesRegex(TypeError, "boolean.*True.*blocked"):
            attention(inputs, inputs, inputs, torch.zeros(3, 3))

    def test_uniform_attention_averages_only_current_and_past_values(self):
        attention = MultiHeadAttention(8, 2, "sinusoidal", dropout_rate=0.0).eval()
        with torch.no_grad():
            attention.q_linear.weight.zero_()
            attention.k_linear.weight.zero_()
            attention.v_linear.weight.copy_(torch.eye(8))
            attention.out_linear.weight.copy_(torch.eye(8))
            attention.out_linear.bias.zero_()
        values = torch.randn(2, 4, 8)
        expected = values.cumsum(dim=1) / torch.arange(1, 5).view(1, 4, 1)
        for model in (self.gpt, self.transformer):
            with self.subTest(model=type(model).__name__):
                mask = model.make_tgt_mask(torch.ones(2, 4, dtype=torch.long))
                torch.testing.assert_close(attention(values, values, values, mask), expected)

    def test_gpt_one_token_and_batched_forward_backward_are_finite(self):
        for batch_size in (1, 2):
            for length in (1, 4):
                with self.subTest(batch=batch_size, length=length):
                    self.gpt.zero_grad(set_to_none=True)
                    target = torch.randint(0, 16, (batch_size, length))
                    logits = self.logits(self.gpt, target)
                    self.assertEqual(logits.shape, (batch_size, length, 16))
                    logits.square().mean().backward()
                    self.assert_finite_gradients(self.gpt)

    def test_transformer_one_token_and_unequal_lengths_forward_backward_are_finite(self):
        for batch_size in (1, 2):
            for source_length, target_length in ((1, 1), (5, 1), (5, 3), (3, 5)):
                with self.subTest(batch=batch_size, source=source_length, target=target_length):
                    self.transformer.zero_grad(set_to_none=True)
                    source = torch.randint(0, 16, (batch_size, source_length))
                    target = torch.randint(0, 16, (batch_size, target_length))
                    logits = self.logits(self.transformer, target, source)
                    self.assertEqual(logits.shape, (batch_size, target_length, 16))
                    logits.square().mean().backward()
                    self.assert_finite_gradients(self.transformer)

    def assert_finite_gradients(self, model):
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.parameters()), 0.0)

    def test_future_tokens_do_not_change_earlier_logits_in_either_model(self):
        for batch_size in (1, 2):
            target = torch.tensor([[1, 2, 3, 4, 5]]).expand(batch_size, -1)
            changed = target.clone()
            changed[:, 3:] = torch.tensor([8, 9])
            for model in (self.gpt, self.transformer):
                with self.subTest(model=type(model).__name__, batch=batch_size), torch.no_grad():
                    source = torch.tensor([[6, 7, 8]]).expand(batch_size, -1) if model is self.transformer else None
                    original_logits = self.logits(model, target, source)
                    changed_logits = self.logits(model, changed, source)
                    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])
                    self.assertGreater((original_logits[:, 3:] - changed_logits[:, 3:]).abs().max().item(), 0.01)

    def test_both_model_families_match_pytorch_causal_attention_reference(self):
        sinusoidal_gpt = GPT(16, 8, num_layers=2, n_heads=2, pos_encoding_type="sinusoidal", dropout_rate=0.0).eval()
        for model in (self.gpt, sinusoidal_gpt, self.transformer):
            for length in (1, 5):
                with self.subTest(model=type(model).__name__, length=length), torch.no_grad():
                    target = torch.randint(0, 16, (2, length))
                    source = torch.randint(0, 16, (2, 3)) if model is self.transformer else None
                    actual = self.logits(model, target, source)
                    with patch.object(MultiHeadAttention, "forward", reference_attention):
                        expected = self.logits(model, target, source)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
