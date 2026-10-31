import math
import unittest

import torch

from blocks import PositionalEncoding


def paper_encoding(pos, i, embed_dim, base=10000.0):
    """Section 3.5 of "Attention Is All You Need", evaluated with scalar math."""
    angle = pos / base ** ((i - i % 2) / embed_dim)
    return math.sin(angle) if i % 2 == 0 else math.cos(angle)


class SinusoidalPositionalEncodingTests(unittest.TestCase):
    def test_matches_paper_formula_for_small_positions_and_widths(self):
        for embed_dim in (2, 4, 6, 16):
            encoding = PositionalEncoding(8, embed_dim).pos_enc[0]
            expected = torch.tensor([[paper_encoding(p, i, embed_dim) for i in range(embed_dim)] for p in range(8)])
            with self.subTest(embed_dim=embed_dim):
                torch.testing.assert_close(encoding, expected, atol=1e-6, rtol=1e-6)

    def test_review_example_width_four_position_one(self):
        # The previous loop gave ~0.99995 here because the pair used a different frequency.
        self.assertAlmostEqual(PositionalEncoding(2, 4).pos_enc[0, 1, 1].item(), math.cos(1.0), places=6)

    def test_long_context_table_is_vectorized_and_correct_far_out(self):
        encoding = PositionalEncoding(16384, 512).pos_enc[0]
        self.assertEqual(encoding.shape, (16384, 512))
        for pos, i in ((16383, 0), (16383, 1), (9000, 510), (9000, 511)):
            self.assertAlmostEqual(encoding[pos, i].item(), paper_encoding(pos, i, 512), places=3)

    def test_rejects_unsupported_dimensions_and_lengths(self):
        for max_seq_len, embed_dim in ((8, 0), (8, 5), (0, 4), (-1, 4)):
            with self.subTest(max_seq_len=max_seq_len, embed_dim=embed_dim), self.assertRaises(ValueError):
                PositionalEncoding(max_seq_len, embed_dim)
        with self.assertRaises(ValueError):
            PositionalEncoding(4, 8)(torch.zeros(1, 5, 8))

    def test_adds_encoding_in_input_dtype_up_to_the_bound(self):
        module = PositionalEncoding(4, 8)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                out = module(torch.zeros(2, 4, 8, dtype=dtype))
                self.assertEqual(out.dtype, dtype)
                torch.testing.assert_close(out[1].float(), module.pos_enc[0].to(dtype).float())

    def test_table_is_not_persisted_and_legacy_tables_are_ignored_on_load(self):
        module = PositionalEncoding(4, 8)
        self.assertEqual(module.state_dict(), {})
        reference = module.pos_enc.clone()
        module.load_state_dict({"pos_enc": torch.ones(1, 4, 8)})  # strict; stale buggy tables must not win
        torch.testing.assert_close(module.pos_enc, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
