import unittest

import torch

from blocks import MultiHeadAttention, RotaryPositionalEncoding
from gpt import GPT


def accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def reference_rope(x, positions, base=10000.0):
    """Independent RoFormer reference: rotate adjacent pairs as complex numbers in FP64."""
    dim = x.size(-1)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    angles = positions.to(torch.float64).unsqueeze(-1) * inv_freq
    pairs = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], dim // 2, 2).contiguous())
    return torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)).flatten(-2)


class RopeConstructionTests(unittest.TestCase):
    def test_rejects_odd_or_non_positive_dimensions(self):
        for dim in (0, -2, 7):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                RotaryPositionalEncoding(dim, 16)
        with self.assertRaises(ValueError):
            RotaryPositionalEncoding(8, 0)

    def test_attention_rejects_odd_rotary_head_dimension(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(embed_dim=6, n_heads=2, rotary=True)

    def test_rejects_invalid_inputs(self):
        rope = RotaryPositionalEncoding(8, 16)
        x = torch.randn(1, 4, 8)
        cases = {
            "wrong feature dim": (ValueError, lambda: rope(torch.randn(1, 4, 6))),
            "2D input": (ValueError, lambda: rope(torch.randn(4, 8))),
            "integer input": (TypeError, lambda: rope(torch.ones(1, 4, 8, dtype=torch.long))),
            "negative offset": (ValueError, lambda: rope(x, offset=-1)),
            "ids and offset": (ValueError, lambda: rope(x, position_ids=torch.arange(4), offset=1)),
            "float ids": (TypeError, lambda: rope(x, position_ids=torch.arange(4.0))),
            "ids length": (ValueError, lambda: rope(x, position_ids=torch.arange(3))),
        }
        for name, (error, call) in cases.items():
            with self.subTest(name), self.assertRaises(error):
                call()

    def test_caches_are_not_saved_in_state_dict(self):
        rope = RotaryPositionalEncoding(8, 16)
        self.assertEqual(rope.state_dict(), {})
        self.assertEqual(rope.cos_cached.dtype, torch.float32)

    def test_loads_checkpoints_that_still_contain_legacy_rope_buffers(self):
        model = GPT(tgt_vocab_size=16, embed_dim=8, num_layers=2, n_heads=2, dropout_rate=0.0)
        state = model.state_dict()
        rope_prefixes = [
            name + "." for name, module in model.named_modules() if isinstance(module, RotaryPositionalEncoding)
        ]
        self.assertEqual(len(rope_prefixes), 2)
        for prefix in rope_prefixes:
            state[prefix + "inv_freq"] = torch.ones(2)
            state[prefix + "cos_cached"] = torch.ones(512, 2)
            state[prefix + "sin_cached"] = torch.zeros(512, 2)
        model.load_state_dict(state)  # strict
        self.assertIn(rope_prefixes[0] + "inv_freq", state, "caller's state dict must not be mutated")


class RopeValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_complex_reference_for_3d_and_4d_inputs(self):
        rope = RotaryPositionalEncoding(8, 32)
        for shape in ((2, 5, 8), (2, 3, 5, 8)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape)
                expected = reference_rope(x, torch.arange(5))
                torch.testing.assert_close(rope(x).double(), expected, atol=1e-5, rtol=1e-5)

    def test_output_dtype_matches_input_and_gradients_are_finite(self):
        rope = RotaryPositionalEncoding(8, 32)
        base = torch.randn(2, 2, 16, 8)
        expected = reference_rope(base, torch.arange(16))
        tolerances = {torch.float32: 1e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}
        for dtype, tol in tolerances.items():
            with self.subTest(dtype=dtype):
                x = base.to(dtype, copy=True).requires_grad_()
                out = rope(x)
                self.assertEqual(out.dtype, dtype)
                torch.testing.assert_close(out.double(), expected, atol=tol, rtol=tol)
                out.float().square().sum().backward()
                self.assertEqual(x.grad.dtype, dtype)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_offset_chunks_match_full_sequence_including_cache_growth(self):
        rope = RotaryPositionalEncoding(8, 4)  # small cache so chunks force extension
        x = torch.randn(2, 2, 11, 8)
        full = rope(x)
        torch.testing.assert_close(full.double(), reference_rope(x, torch.arange(11)), atol=1e-5, rtol=1e-5)

        fresh = RotaryPositionalEncoding(8, 4)
        bounds = [0, 3, 4, 9, 10, 11]
        chunks = [fresh(x[:, :, start:end], offset=start) for start, end in zip(bounds, bounds[1:])]
        torch.testing.assert_close(torch.cat(chunks, dim=2), full, atol=0, rtol=0)
        self.assertGreaterEqual(fresh.cos_cached.size(0), 11)

    def test_position_ids_match_offsets_and_support_per_row_positions(self):
        rope = RotaryPositionalEncoding(8, 4)
        x = torch.randn(2, 2, 3, 8)
        torch.testing.assert_close(rope(x, position_ids=torch.arange(5, 8)), rope(x, offset=5), atol=0, rtol=0)

        rows = torch.tensor([[0, 1, 2], [6, 7, 8]])
        out = rope(x, position_ids=rows)
        for row in range(2):
            torch.testing.assert_close(out[row], rope(x[row : row + 1], offset=int(rows[row, 0]))[0], atol=0, rtol=0)

        x3 = torch.randn(2, 3, 8)
        out3 = rope(x3, position_ids=rows)
        torch.testing.assert_close(out3[1], rope(x3[1:], offset=6)[0], atol=0, rtol=0)

    def test_cache_grown_under_inference_mode_still_supports_training(self):
        rope = RotaryPositionalEncoding(8, 4)
        with torch.inference_mode():
            rope(torch.randn(1, 8, 8))  # grows the cache
        self.assertFalse(rope.cos_cached.is_inference())
        x = torch.randn(1, 8, 8, requires_grad=True)
        rope(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_module_half_cast_does_not_degrade_cached_angles(self):
        x = torch.randn(1, 2, 40, 8).half()
        expected = RotaryPositionalEncoding(8, 64)(x)
        rope = RotaryPositionalEncoding(8, 64).half()
        out = rope(x)
        self.assertEqual(rope.cos_cached.dtype, torch.float32)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


class RopeDeviceTests(unittest.TestCase):
    def test_cache_follows_input_device_and_extension_stays_there(self):
        rope = RotaryPositionalEncoding(8, 4)
        meta = torch.empty(1, 2, 3, 8, device="meta")
        out = rope(meta, offset=6)  # needs extension and a device change at once
        self.assertEqual(out.device.type, "meta")
        self.assertEqual(rope.cos_cached.device.type, "meta")
        self.assertEqual(rope.sin_cached.device.type, "meta")

        x = torch.randn(1, 2, 3, 8)
        cpu_out = rope(x, offset=6)
        self.assertEqual(rope.cos_cached.device.type, "cpu")
        torch.testing.assert_close(cpu_out, RotaryPositionalEncoding(8, 16)(x, offset=6), atol=0, rtol=0)

    @unittest.skipIf(accelerator() is None, "requires CUDA or MPS")
    def test_accelerator_migration_extension_and_low_precision(self):
        device = accelerator()
        torch.manual_seed(0)
        x = torch.randn(2, 2, 12, 8)
        expected = reference_rope(x, torch.arange(4, 16))

        rope = RotaryPositionalEncoding(8, 4).to(device)
        for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 4e-3), (torch.bfloat16, 3e-2)):
            with self.subTest(dtype=dtype):
                inputs = x.to(device, dtype).requires_grad_()
                out = rope(inputs, offset=4)
                self.assertEqual((out.device.type, out.dtype), (device.type, dtype))
                self.assertEqual(rope.cos_cached.device.type, device.type)
                torch.testing.assert_close(out.cpu().double(), expected, atol=tol, rtol=tol)
                out.float().square().sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

        rope.cpu()
        self.assertEqual(rope.cos_cached.device.type, "cpu")
        self.assertEqual(rope(x, offset=4).device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_autocast_attention_has_finite_gradients(self):
        dtypes = [torch.float16] + ([torch.bfloat16] if torch.cuda.is_bf16_supported() else [])
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                attention = MultiHeadAttention(32, 4, rotary=True, max_seq_len=8, dropout_rate=0.0).cuda()
                x = torch.randn(2, 24, 32, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=dtype):
                    out = attention(x, x, x, is_causal=True)
                self.assertEqual(out.dtype, dtype)
                out.float().square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in attention.parameters()))


if __name__ == "__main__":
    unittest.main()
