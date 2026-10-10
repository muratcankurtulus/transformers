import argparse
import io
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from blocks import RotaryPositionalEncoding
from gpt import GPT
from tokenizer import Tokenizer
from train_budget import PackedTokens, train


def write_tokens(path, tokens, tokenizer):
    array = np.array(tokens, dtype="<u2")
    path.write_bytes(b"BPE1" + bytes([2]) + struct.pack("<Q", len(array)) + array.tobytes())
    tokenizer.save_metadata(str(path) + ".tokenizer.json")


class CausalAndPrecisionTests(unittest.TestCase):
    def test_changing_future_tokens_does_not_change_past_logits(self):
        torch.manual_seed(0)
        model = GPT(tgt_vocab_size=260, embed_dim=16, num_layers=1, n_heads=2, dropout_rate=0.0).eval()
        original = torch.tensor([[10, 20, 30, 40, 50]])
        changed = torch.tensor([[10, 20, 90, 100, 110]])
        implicit = model(original)
        explicit = model(original, model.make_tgt_mask(original))
        torch.testing.assert_close(implicit, explicit)
        torch.testing.assert_close(implicit[:, :2], model(changed)[:, :2])

    def test_rope_preserves_half_dtype_and_matches_float_reference(self):
        rope = RotaryPositionalEncoding(8, 32)
        values = torch.randn(2, 2, 16, 8)
        result = rope(values.half())
        self.assertEqual(result.dtype, torch.float16)
        torch.testing.assert_close(result.float(), rope(values), atol=0.004, rtol=0.004)

    def test_generation_bounds_context_and_preserves_prompt(self):
        model = GPT(tgt_vocab_size=260, embed_dim=16, num_layers=1, n_heads=2, dropout_rate=0.0).eval()
        prompt = [10, 20, 30, 40, 50]
        output = model.generate(prompt, 3, context_length=2)
        self.assertEqual(output.shape, (1, 8))
        self.assertEqual(output[0, :5].tolist(), prompt)


class PackedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.tokenizer = Tokenizer(vocab_size=260)
        self.tokenizer_dir = self.directory / "tokenizer"
        self.tokenizer.save(self.tokenizer_dir)
        self.train_path = self.directory / "train.bin"
        self.eval_path = self.directory / "eval.bin"
        write_tokens(self.train_path, list(range(100)), self.tokenizer)
        write_tokens(self.eval_path, list(range(100, 200)), self.tokenizer)

    def test_nonoverlapping_contexts_use_the_correct_next_token_targets(self):
        data = PackedTokens(self.train_path, 8, self.tokenizer)
        self.assertIsInstance(data.data, np.memmap)
        self.assertEqual(data.num_blocks, 12)
        inputs, targets = data.batch([0, 1])
        self.assertEqual(inputs.tolist(), [list(range(8)), list(range(8, 16))])
        self.assertEqual(targets.tolist(), [list(range(1, 9)), list(range(9, 17))])

    def test_corrupt_token_count_is_rejected(self):
        self.train_path.write_bytes(self.train_path.read_bytes() + b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "file size"):
            PackedTokens(self.train_path, 8, self.tokenizer)

    def arguments(self):
        return argparse.Namespace(
            device="cpu",
            precision="fp32",
            resume=None,
            output_dir=self.directory / "run",
            tokenizer=self.tokenizer_dir,
            train_data=self.train_path,
            eval_data=self.eval_path,
            seq_len=8,
            embed_dim=16,
            num_layers=1,
            n_heads=2,
            dropout_rate=0.0,
            batch_size=2,
            gradient_accumulation=2,
            learning_rate=1e-3,
            weight_decay=0.1,
            warmup_steps=0,
            seed=42,
            max_hours=1,
            max_steps=1,
            log_interval=1,
            eval_interval=10,
            eval_batches=1,
            checkpoint_seconds=600,
        )

    def test_training_checkpoint_resumes_weights_optimizer_rng_and_token_count(self):
        args = self.arguments()
        with redirect_stdout(io.StringIO()):
            train(args)
        last = args.output_dir / "last.pth"
        before = torch.load(last, weights_only=True)
        self.assertEqual(before["tokens_seen"], 32)
        self.assertEqual(before["model_config"]["tgt_vocab_size"], 260)
        args.resume, args.max_steps = last, 2
        with redirect_stdout(io.StringIO()):
            train(args)
        after = torch.load(last, weights_only=True)
        self.assertEqual(after["step"], 2)
        self.assertEqual(after["tokens_seen"], 64)
        self.assertFalse(torch.equal(before["sampling_rng"], after["sampling_rng"]))
        self.assertTrue(
            any(
                not torch.equal(value, after["model_state_dict"][name])
                for name, value in before["model_state_dict"].items()
            )
        )
        write_tokens(self.eval_path, list(range(90, 190)), self.tokenizer)
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            train(args)


if __name__ == "__main__":
    unittest.main()
