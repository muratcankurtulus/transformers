import io
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from blocks import MultiHeadAttention, PositionalEncoding
from blocks import RotaryPositionalEncoding as RoPE
from gpt import GPT
from transformer import Transformer

MODES = ("rotary", "sinusoidal")


def build(kind, mode, dropout_rate=0.0, length=16):
    if kind == "gpt":
        return GPT(
            16, 8, num_layers=2, n_heads=2, pos_encoding_type=mode, dropout_rate=dropout_rate, max_seq_len=length
        )
    return Transformer(8, 16, 16, length, num_layers=2, n_heads=2, pos_encoding_type=mode, dropout_rate=dropout_rate)


def run(model):
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    if isinstance(model, GPT):
        return model(tokens)
    return model(tokens, tokens[:, :3])


def cross_attentions(model):
    if isinstance(model, GPT):
        return set()
    return {id(layer.encoder_block.attention) for layer in model.decoder.layers}


class PositionalModeTests(unittest.TestCase):
    def test_modes_are_exclusive_in_every_stack(self):
        for kind in ("gpt", "transformer"):
            for mode in MODES:
                with self.subTest(kind=kind, mode=mode):
                    model = build(kind, mode)
                    modules = list(model.modules())
                    tables = [m for m in modules if isinstance(m, PositionalEncoding)]
                    ropes = [m for m in modules if isinstance(m, RoPE)]
                    cross = cross_attentions(model)
                    self_attentions = [m for m in modules if isinstance(m, MultiHeadAttention) and id(m) not in cross]
                    stacks = 1 if kind == "gpt" else 2
                    if mode == "rotary":
                        self.assertEqual(tables, [])
                        self.assertTrue(all(attention.rotary for attention in self_attentions))
                        self.assertEqual(len(ropes), len(self_attentions))
                    else:
                        self.assertEqual(len(tables), stacks)
                        self.assertEqual(ropes, [])
                        self.assertFalse(any(attention.rotary for attention in self_attentions))

    def test_cross_attention_never_rotates(self):
        for mode in MODES:
            model = build("transformer", mode)
            for layer in model.decoder.layers:
                with self.subTest(mode=mode):
                    self.assertFalse(layer.encoder_block.attention.rotary)
                    self.assertFalse(hasattr(layer.encoder_block.attention, "rope"))

    def test_forward_runs_only_the_selected_encoding(self):
        forbidden = {"rotary": PositionalEncoding, "sinusoidal": RoPE}
        required = {"rotary": RoPE, "sinusoidal": PositionalEncoding}
        for kind in ("gpt", "transformer"):
            for mode in MODES:
                with self.subTest(kind=kind, mode=mode):
                    model = build(kind, mode).eval()
                    used = required[mode].forward
                    with (
                        patch.object(forbidden[mode], "forward", side_effect=AssertionError("wrong encoding ran")),
                        patch.object(required[mode], "forward", autospec=True, side_effect=used) as spy,
                    ):
                        out = run(model)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertGreater(spy.call_count, 0)

    def test_sinusoidal_gpt_is_position_sensitive(self):
        # Previously sinusoidal GPT had no positional signal: logits at a position ignored token order.
        torch.manual_seed(0)
        model = build("gpt", "sinusoidal").eval()
        a = model(torch.tensor([[3, 5, 7, 9]]))[0, -1]
        b = model(torch.tensor([[7, 5, 3, 9]]))[0, -1]
        self.assertFalse(torch.allclose(a, b))

    def test_rejects_unknown_modes(self):
        for kind in ("gpt", "transformer"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build(kind, "learned")

    def test_dropout_and_lengths_reach_every_nested_block(self):
        for kind in ("gpt", "transformer"):
            for mode in MODES:
                with self.subTest(kind=kind, mode=mode):
                    model = build(kind, mode, dropout_rate=0.37, length=24)
                    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
                    self.assertTrue(dropouts)
                    self.assertEqual({m.p for m in dropouts}, {0.37})
                    for module in model.modules():
                        if isinstance(module, (PositionalEncoding, RoPE)):
                            self.assertEqual(module.max_seq_len, 24)

    def test_configuration_survives_save_and_load(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(0)
                config = {"tgt_vocab_size": 16, "embed_dim": 8, "num_layers": 2, "n_heads": 2}
                config |= {"dropout_rate": 0.0, "pos_encoding_type": mode, "max_seq_len": 16}
                model = GPT(**config).eval()
                buffer = io.BytesIO()
                torch.save({"model_config": config, "model_state_dict": model.state_dict()}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                restored = GPT(**saved["model_config"]).eval()
                restored.load_state_dict(saved["model_state_dict"])
                self.assertEqual(restored.decoder.pos_encoding_type, mode)
                tokens = torch.tensor([[1, 2, 3, 4]])
                torch.testing.assert_close(restored(tokens), model(tokens), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
