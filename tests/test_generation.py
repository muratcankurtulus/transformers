import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import generate
import train_gpt
from generation import decode, select_next_token
from gpt import GPT
from tokenizer import Tokenizer
from train_gpt import save_checkpoint
from transformer import Transformer

VOCAB = 8
EOS = 5


def successor_logits(history):
    """Scripted next-token logits: always predict (last token + 1) mod VOCAB."""
    return torch.nn.functional.one_hot((history[:, -1] + 1) % VOCAB, VOCAB).float()


def tiny_gpt(**overrides):
    config = {"tgt_vocab_size": VOCAB, "embed_dim": 8, "num_layers": 1, "n_heads": 2, "dropout_rate": 0.0}
    return GPT(**(config | overrides))


def tiny_transformer(seq_len=12, **overrides):
    config = {"num_layers": 2, "n_heads": 2, "dropout_rate": 0.0} | overrides
    return Transformer(8, 11, VOCAB, seq_len, **config)


class DecodeLoopTests(unittest.TestCase):
    def test_each_row_stops_at_its_own_eos_and_is_padded_with_eos(self):
        prompts = torch.tensor([[3], [0], [4]])

        output = decode(successor_logits, prompts, 10, eos_token_id=EOS)

        # Row 2 finishes first, row 1 next; the loop ends once the slowest row (0 -> ... -> 5) finishes.
        self.assertEqual(output.tolist(), [[3, 4, 5, 5, 5, 5], [0, 1, 2, 3, 4, 5], [4, 5, 5, 5, 5, 5]])

    def test_without_eos_the_full_budget_is_generated(self):
        output = decode(successor_logits, torch.tensor([[3]]), 4)

        self.assertEqual(output.tolist(), [[3, 4, 5, 6, 7]])

    def test_budget_bounds_rows_that_never_reach_eos(self):
        output = decode(successor_logits, torch.tensor([[4], [6]]), 3, eos_token_id=EOS)

        self.assertEqual(output.tolist(), [[4, 5, 5, 5], [6, 7, 0, 1]])

    def test_stops_calling_the_model_once_every_row_is_finished(self):
        calls = []

        def counted(history):
            calls.append(history.size(1))
            return successor_logits(history)

        decode(counted, torch.tensor([[3], [4]]), 50, eos_token_id=EOS)

        self.assertEqual(calls, [1, 2])

    def test_zero_budget_returns_the_prompt(self):
        prompt = torch.tensor([[1, 2, 3]])

        self.assertTrue(torch.equal(decode(successor_logits, prompt, 0, eos_token_id=EOS), prompt))


class GPTGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def scripted(self, model, seen_windows=None):
        def forward(x, mask=None):
            if seen_windows is not None:
                seen_windows.append(x.tolist())
            logits = successor_logits(x)
            return logits.unsqueeze(1).expand(x.size(0), x.size(1), VOCAB)

        return patch.object(model, "forward", forward)

    def test_single_and_batched_prompts_stop_per_sequence_at_eos(self):
        model = tiny_gpt()
        with self.scripted(model):
            single = model.generate([2, 3], 10, eos_token_id=EOS)
            batched = model.generate([[2, 3], [0, 1], [3, 4]], 10, eos_token_id=EOS)

        self.assertEqual(single.tolist(), [[2, 3, 4, 5]])
        self.assertEqual(batched.tolist(), [[2, 3, 4, 5, 5, 5], [0, 1, 2, 3, 4, 5], [3, 4, 5, 5, 5, 5]])

    def test_max_new_tokens_bounds_single_and_batched_generation(self):
        model = tiny_gpt().eval()
        prompts = torch.tensor([[1, 2], [3, 4]])

        batched = model.generate(prompts, 4)
        rows = [model.generate(prompt.tolist(), 4) for prompt in prompts]

        self.assertEqual(batched.shape, (2, 6))
        self.assertTrue(torch.equal(batched[:, :2], prompts))
        # Greedy rows are independent, so batching must not change any sequence.
        self.assertTrue(torch.equal(batched, torch.cat(rows)))
        self.assertEqual(model.generate([1], 0).tolist(), [[1]])

    def test_sliding_window_conditions_on_the_most_recent_tokens(self):
        model = tiny_gpt(max_seq_len=4)
        seen = []
        with self.scripted(model, seen):
            output = model.generate([1, 2, 3, 4, 5, 6], 3)

        self.assertEqual(output.tolist(), [[1, 2, 3, 4, 5, 6, 7, 0, 1]])
        self.assertEqual(seen, [[[3, 4, 5, 6]], [[4, 5, 6, 7]], [[5, 6, 7, 0]]])

    def test_explicit_context_length_narrows_the_window(self):
        model = tiny_gpt(max_seq_len=4)
        seen = []
        with self.scripted(model, seen):
            model.generate([1, 2, 3], 2, context_length=np.int64(2))

        self.assertEqual(seen, [[[2, 3]], [[3, 4]]])

    def test_sinusoidal_model_generates_past_its_table_by_sliding(self):
        model = tiny_gpt(pos_encoding_type="sinusoidal", max_seq_len=4)

        output = model.generate([1, 2, 3], 6, temperature=1.0, top_k=3)

        self.assertEqual(output.shape, (1, 9))

    def test_eos_stopping_works_with_a_real_forward_pass(self):
        model = tiny_gpt()
        with torch.no_grad():
            model.decoder.fully_connected.weight.zero_()
            model.decoder.fully_connected.bias.zero_()
            model.decoder.fully_connected.bias[EOS] = 1.0

        output = model.generate([[1, 2], [3, 4]], 10, eos_token_id=EOS)

        self.assertEqual(output.tolist(), [[1, 2, EOS], [3, 4, EOS]])

    def test_generation_runs_in_eval_mode_and_restores_the_callers_mode(self):
        for training in (True, False):
            with self.subTest(training=training):
                model = tiny_gpt().train(training)
                modes = []

                def forward(x, mask=None):
                    modes.append(model.training)
                    raise RuntimeError("forward failed")

                with patch.object(model, "forward", forward), self.assertRaisesRegex(RuntimeError, "forward failed"):
                    model.generate([1, 2], 2)
                self.assertEqual(modes, [False])
                self.assertEqual(model.training, training)
                model.generate([1, 2], 2)
                self.assertEqual(model.training, training)

    def test_numpy_integers_are_accepted(self):
        model = tiny_gpt().eval()

        output = model.generate(np.array([1, 2]), np.int64(2), temperature=1.0, top_k=np.int64(3))

        self.assertEqual(output.shape, (1, 4))
        self.assertTrue(torch.equal(model.generate(np.array([[1, 2]]), 2), model.generate([1, 2], 2)))

    def test_output_never_aliases_the_prompt(self):
        model = tiny_gpt()
        for prompt in (torch.tensor([1, 2]), torch.tensor([[1, 2]])):
            with self.subTest(shape=tuple(prompt.shape)):
                output = model.generate(prompt, 0)
                output.fill_(7)
                self.assertEqual(prompt.flatten().tolist(), [1, 2])

    def test_forbidden_tokens_do_not_modify_the_callers_logits(self):
        logits = torch.zeros(1, VOCAB)
        logits[0, 3] = 1.0

        token = select_next_token(logits, forbidden_token_ids=[3])

        self.assertNotEqual(token.item(), 3)
        self.assertEqual(logits[0, 3].item(), 1.0)

    def test_invalid_prompts_raise_informative_errors(self):
        model = tiny_gpt()
        cases = {
            "empty list": ([], ValueError, "at least one token"),
            "empty row": ([[]], ValueError, "at least one token"),
            "empty tensor": (torch.empty(2, 0, dtype=torch.long), ValueError, "at least one token"),
            "ragged batch": ([[1, 2], [3]], ValueError, "padding masks"),
            "3-D tensor": (torch.ones(1, 2, 3, dtype=torch.long), ValueError, r"\[batch, length\]"),
            "float IDs": (torch.tensor([1.0, 2.0]), TypeError, "integers"),
            "too large ID": ([1, VOCAB], ValueError, r"\[0, 8\)"),
            "negative ID": ([-1, 2], ValueError, r"\[0, 8\)"),
            "bool IDs": (torch.tensor([True, False]), TypeError, "integers"),
            "text prompt": ("ab", ValueError, "integer token IDs"),
        }
        for label, (prompt, error, message) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(error, message):
                model.generate(prompt, 1)

    def test_invalid_settings_raise_informative_errors(self):
        model = tiny_gpt(max_seq_len=4)
        cases = {
            "context past trained length": ({"context_length": 5}, "max_seq_len=4"),
            "empty context": ({"context_length": 0}, "context_length"),
            "negative budget": ({"max_new_tokens": -1}, "max_new_tokens"),
            "eos forbidden": ({"eos_token_id": EOS, "forbidden_token_ids": [EOS]}, "never stop"),
            "eos outside vocabulary": ({"eos_token_id": VOCAB}, "outside the vocabulary"),
            "zero top_p": ({"top_p": 0.0}, "top_p"),
            "negative temperature": ({"temperature": -1.0}, "temperature"),
            "NaN temperature": ({"temperature": float("nan")}, "temperature must be finite"),
            "everything forbidden": ({"forbidden_token_ids": list(range(VOCAB))}, "whole vocabulary"),
        }
        for label, (overrides, message) in cases.items():
            kwargs = {"max_new_tokens": 1} | overrides
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                model.generate([1, 2], **kwargs)

    def test_non_integer_counts_raise_type_errors(self):
        model = tiny_gpt(max_seq_len=4)
        cases = {
            "bool budget": {"max_new_tokens": True},
            "float budget": {"max_new_tokens": 2.0},
            "float context": {"max_new_tokens": 1, "context_length": 2.0},
            "float top_k": {"max_new_tokens": 1, "temperature": 1.0, "top_k": 2.5},
        }
        for label, kwargs in cases.items():
            with self.subTest(label), self.assertRaisesRegex(TypeError, "must be an integer"):
                model.generate([1, 2], **kwargs)


class TransformerGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.src = torch.randint(0, 11, (3, 5))
        self.start = torch.ones(3, 1, dtype=torch.long)

    def reference(self, model, src, tgt, steps):
        """Greedy decoding by rerunning the full forward pass on the whole history each step."""
        model.eval()
        with torch.no_grad():
            for _ in range(steps):
                tgt = torch.cat((tgt, model(src, tgt)[:, -1, :].argmax(-1, keepdim=True)), dim=1)
        return tgt

    def test_batched_greedy_generation_matches_full_history_reference(self):
        for mode in ("rotary", "sinusoidal"):
            with self.subTest(mode=mode):
                model = tiny_transformer(pos_encoding_type=mode)

                output = model.generate(self.src, self.start, 6)

                self.assertEqual(output.shape, (3, 7))
                self.assertTrue(torch.equal(output, self.reference(model, self.src, self.start, 6)))
                rows = [model.generate(self.src[i], self.start[i], 6) for i in range(3)]
                self.assertTrue(torch.equal(output, torch.cat(rows)))

    def test_encoder_runs_once_and_decoder_sees_growing_history_with_matching_masks(self):
        model = tiny_transformer()
        encoder_calls, decoder_inputs = [], []
        model.encoder.register_forward_hook(lambda _m, inputs, _out: encoder_calls.append(inputs[0].shape))
        model.decoder.register_forward_hook(
            lambda _m, inputs, _out: decoder_inputs.append((inputs[0].shape, inputs[2].shape))
        )

        model.generate(self.src, torch.ones(3, 2, dtype=torch.long), 3)

        self.assertEqual(encoder_calls, [(3, 5)])
        self.assertEqual(
            decoder_inputs,
            [((3, n), (3, 1, n, n)) for n in (2, 3, 4)],
        )

    def test_rows_stop_independently_at_eos(self):
        model = tiny_transformer()

        def decoder(tgt, _enc_out, _mask):
            return successor_logits(tgt).unsqueeze(1).expand(tgt.size(0), tgt.size(1), VOCAB)

        with patch.object(model.decoder, "forward", decoder):
            output = model.generate(self.src, [[3], [0], [4]], 10, eos_token_id=EOS)

        self.assertEqual(output.tolist(), [[3, 4, 5, 5, 5, 5], [0, 1, 2, 3, 4, 5], [4, 5, 5, 5, 5, 5]])

    def test_output_budget_is_max_new_tokens_not_source_length(self):
        model = tiny_transformer()

        self.assertEqual(model.generate(self.src, self.start, 2).shape, (3, 3))
        self.assertEqual(model.generate(self.src[:, :2], self.start, 8).shape, (3, 9))

    def test_full_trained_context_can_be_generated(self):
        # The final token is never fed back, so one start token plus seq_len new tokens fits exactly.
        for mode in ("rotary", "sinusoidal"):
            with self.subTest(mode=mode):
                model = tiny_transformer(seq_len=6, pos_encoding_type=mode)
                lengths = []
                model.decoder.register_forward_hook(lambda _m, inputs, _out: lengths.append(inputs[0].size(1)))

                output = model.generate(self.src, self.start, 6)

                self.assertEqual(output.shape, (3, 7))
                self.assertEqual(max(lengths), 6)
                self.assertTrue(torch.equal(output, self.reference(model, self.src, self.start, 6)))

    def test_generation_runs_in_eval_mode_and_restores_the_callers_mode(self):
        model = tiny_transformer().train()
        modes = []

        def decoder(tgt, _enc_out, _mask):
            modes.append(model.training)
            raise RuntimeError("decoder failed")

        with patch.object(model.decoder, "forward", decoder), self.assertRaisesRegex(RuntimeError, "decoder failed"):
            model.generate(self.src, self.start, 2)
        self.assertEqual(modes, [False])
        self.assertTrue(model.training)
        model.generate(self.src, self.start, 2)
        self.assertTrue(model.training)

    def test_invalid_inputs_and_context_overflow_raise_informative_errors(self):
        model = tiny_transformer(seq_len=6)
        cases = {
            "batch mismatch": (self.src, torch.ones(2, 1, dtype=torch.long), 1, "batch sizes differ"),
            "source too long": (torch.ones(3, 7, dtype=torch.long), self.start, 1, "src length 7 exceeds"),
            "target overflow": (self.src, self.start, 7, "needs 7 decoder positions, exceeding seq_len 6"),
            "empty target": (self.src, torch.empty(3, 0, dtype=torch.long), 1, "tgt must contain"),
            "empty source": ([], self.start[:1], 1, "src must contain"),
            "target ID outside vocabulary": (self.src, [[VOCAB]] * 3, 1, r"tgt token IDs must be in \[0, 8\)"),
        }
        for label, (src, tgt, budget, message) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                model.generate(src, tgt, budget)


class GenerateCliTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.tokenizer = Tokenizer(vocab_size=260)
        self.tokenizer.save(self.directory / "tokenizer")
        self.config = {"tgt_vocab_size": 260, "embed_dim": 16, "num_layers": 1, "n_heads": 2}
        self.config |= {"expansion_factor": 4, "dropout_rate": 0.0, "pos_encoding_type": "sinusoidal"}
        self.config["max_seq_len"] = 8
        self.model = GPT(**self.config)
        self.checkpoint = self.directory / "model.pth"
        self.calls = []

    def run_cli(self, *extra, checkpoint=None):
        """Run generate.py and return its stdout; each GPT.generate call is recorded in self.calls."""
        argv = ["generate.py", "--prompt", "ab", "--model_path", str(checkpoint or self.checkpoint)]
        argv += ["--tokenizer_path", str(self.directory / "tokenizer"), "--device", "cpu", "--temperature", "0"]
        original = GPT.generate

        def recording(model, *args, **kwargs):
            output = original(model, *args, **kwargs)
            self.calls.append((args, kwargs, output))
            return output

        stdout = io.StringIO()
        with patch.object(sys, "argv", argv + list(extra)), patch.object(GPT, "generate", recording):
            with redirect_stdout(stdout):
                generate.main()
        return stdout.getvalue()

    def force_eos_logits(self):
        """Make every next-token logit 0 except EOS (1.0), so greedy output is fully determined."""
        with torch.no_grad():
            self.model.decoder.fully_connected.weight.zero_()
            self.model.decoder.fully_connected.bias.zero_()
            self.model.decoder.fully_connected.bias[Tokenizer.SPECIAL_TOKENS["<EOS>"]] = 1.0

    def test_saved_config_reconstructs_without_architecture_flags(self):
        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)

        self.assertEqual(torch.load(self.checkpoint, weights_only=True)["model_config"], self.config)
        # 2 + 12 tokens overflow the 8-entry sinusoidal table unless the window comes from max_seq_len.
        self.assertTrue(self.run_cli("--max_new_tokens", "12").startswith("ab"))
        ((_, kwargs, output),) = self.calls
        self.assertEqual(kwargs["context_length"], 8)
        self.assertEqual(output.shape, (1, 14))

    def test_stop_at_eos_allows_and_stops_on_eos(self):
        eos = Tokenizer.SPECIAL_TOKENS["<EOS>"]
        self.force_eos_logits()
        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)

        self.assertEqual(self.run_cli("--stop_at_eos", "--max_new_tokens", "5"), "ab\n")
        ((_, kwargs, output),) = self.calls
        self.assertEqual(kwargs["eos_token_id"], eos)
        self.assertNotIn(eos, kwargs["forbidden_token_ids"])
        self.assertEqual(output.tolist(), [[ord("a"), ord("b"), eos]])

    def test_eos_stays_forbidden_by_default_and_length_is_an_alias(self):
        self.force_eos_logits()
        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)

        # With every special ID forbidden, the tied zero logits make greedy decoding pick byte 0.
        self.assertEqual(self.run_cli("--length", "5"), "ab" + "\x00" * 5 + "\n")
        ((args, kwargs, output),) = self.calls
        self.assertEqual(args[1], 5)
        self.assertIsNone(kwargs["eos_token_id"])
        self.assertEqual(output.shape, (1, 7))

    def test_checkpoint_without_config_requires_flags_and_rejects_oversized_context(self):
        legacy = self.directory / "legacy.pth"
        torch.save({"model_state_dict": self.model.state_dict(), "tokenizer": self.tokenizer.metadata}, legacy)
        flags = ["--embed_dim", "16", "--n_heads", "2", "--num_layers", "1", "--expansion_factor", "4"]
        flags += ["--tgt_vocab_size", "260", "--dropout_rate", "0", "--pos_encoding_type", "sinusoidal"]
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.run_cli(checkpoint=legacy)
        self.assertIn("supply --embed_dim", stderr.getvalue())

        self.assertTrue(self.run_cli(*flags, "--seq_len", "8", checkpoint=legacy).startswith("ab"))
        self.assertEqual(self.calls[-1][1]["context_length"], 8)

        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)
        self.run_cli("--seq_len", "4")
        self.assertEqual(self.calls[-1][1]["context_length"], 4)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.run_cli("--seq_len", "16")
        self.assertIn("exceeds the checkpoint's trained context (8 tokens)", stderr.getvalue())

    def test_train_gpt_checkpoint_generates_without_architecture_flags(self):
        text = "the quick brown fox jumps over the lazy dog. " * 4
        for split in ("train", "eval"):
            (self.directory / f"{split}.txt").write_text(text, encoding="utf-8")
        real_loader = torch.utils.data.DataLoader

        def single_process_loader(*args, **kwargs):
            return real_loader(*args, **(kwargs | {"num_workers": 0}))

        run = self.directory / "run"
        with patch("torch.utils.data.DataLoader", single_process_loader):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                train_gpt.main(
                    tokenizer_path=str(self.directory / "tokenizer"),
                    train_data_path=str(self.directory / "train.txt"),
                    eval_data_path=str(self.directory / "eval.txt"),
                    epochs=1,
                    experiment_name=str(run),
                    tokenizer_type="default",
                    eval_interval=1000,
                    use_pretokenized=False,
                    weight_decay=0.0,
                    early_stopping_patience=1,
                    dropout_rate=0.0,
                    embed_dim=16,
                    tgt_vocab_size=260,
                    seq_len=6,
                    num_layers=1,
                    expansion_factor=4,
                    n_heads=2,
                    batch_size=16,
                    shuffle=False,
                    pos_encoding_type="sinusoidal",
                    device="cpu",
                )

        best = Path(f"{run}_best.pth")
        saved = torch.load(best, weights_only=True)
        self.assertEqual(saved["model_config"]["max_seq_len"], 6)
        self.assertEqual(saved["model_config"]["pos_encoding_type"], "sinusoidal")
        self.assertTrue(self.run_cli("--max_new_tokens", "10", checkpoint=best).startswith("ab"))
        self.assertEqual(self.calls[-1][1]["context_length"], 6)


if __name__ == "__main__":
    unittest.main()
