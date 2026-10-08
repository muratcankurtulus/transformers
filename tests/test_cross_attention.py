import unittest

import torch

from blocks import TransformerDecoderBlock, TransformerEncoderBlock


class CrossAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_length_follows_queries_with_unequal_source_and_target_lengths(self):
        block = TransformerEncoderBlock(embed_dim=8, n_heads=2, pos_encoding_type="sinusoidal", dropout_rate=0.0).eval()

        for source_length, target_length in ((5, 3), (3, 5)):
            with self.subTest(source_length=source_length, target_length=target_length):
                source = torch.randn(2, source_length, 8)
                target = torch.randn(2, target_length, 8)

                output = block(key=source, query=target, value=source)

                self.assertEqual(output.shape, target.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_reordering_source_keys_and_values_preserves_equal_length_target_outputs(self):
        # Without positional rotation, reordering memory must leave each fixed query's result unchanged.
        block = TransformerEncoderBlock(embed_dim=8, n_heads=2, pos_encoding_type="sinusoidal", dropout_rate=0.0).eval()
        source = torch.randn(2, 4, 8)
        target = torch.randn(2, 4, 8)
        reordered_source = source[:, [2, 0, 3, 1], :]

        output = block(key=source, query=target, value=source)
        reordered_output = block(key=reordered_source, query=target, value=reordered_source)

        torch.testing.assert_close(reordered_output, output, rtol=1e-5, atol=1e-6)

    def test_decoder_cross_attention_supports_unequal_lengths_and_backpropagation(self):
        block = TransformerDecoderBlock(embed_dim=8, n_heads=2, dropout_rate=0.0).eval()

        for source_length, target_length in ((5, 3), (3, 5)):
            with self.subTest(source_length=source_length, target_length=target_length):
                block.zero_grad(set_to_none=True)
                source = torch.randn(2, source_length, 8, requires_grad=True)
                target = torch.randn(2, target_length, 8, requires_grad=True)
                # Supply a valid mask directly; the model's mask helper is a separate review item.
                mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool), diagonal=1)

                output = block(key=source, x=target, value=source, mask=mask)

                self.assertEqual(output.shape, target.shape)
                self.assertTrue(torch.isfinite(output).all())
                output[0, 0, 0].backward()
                for inputs in (source, target):
                    self.assertIsNotNone(inputs.grad)
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    self.assertGreater(inputs.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
