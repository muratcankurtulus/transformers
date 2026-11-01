import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import generate
from generation import decode
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

    def scripted(self, model, seen_lengths=None):
        def forward(x, mask=None):
            if seen_lengths is not None:
                seen_lengths.append(x.size(1))
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

    def test_sliding_window_never_exceeds_the_context_length(self):
        model = tiny_gpt(max_seq_len=4)
        seen = []
        with self.scripted(model, seen):
            output = model.generate([1, 2, 3, 4, 5, 6], 3)

        self.assertEqual(output.shape, (1, 9))
        self.assertEqual(seen, [4, 4, 4])

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

    def test_generation_restores_training_mode(self):
        model = tiny_gpt().train()

        model.generate([1, 2], 2)

        self.assertTrue(model.training)

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
        }
        for label, (overrides, message) in cases.items():
            kwargs = {"max_new_tokens": 1} | overrides
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
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

    def test_invalid_inputs_and_context_overflow_raise_informative_errors(self):
        model = tiny_transformer(seq_len=6)
        cases = {
            "batch mismatch": (self.src, torch.ones(2, 1, dtype=torch.long), 1, "batch sizes differ"),
            "source too long": (torch.ones(3, 7, dtype=torch.long), self.start, 1, "src length 7 exceeds"),
            "target overflow": (self.src, self.start, 6, "exceeds seq_len 6"),
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

    def run_cli(self, *extra):
        argv = ["generate.py", "--prompt", "ab", "--model_path", str(self.checkpoint), "--device", "cpu"]
        argv += ["--tokenizer_path", str(self.directory / "tokenizer"), "--temperature", "0", *extra]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            generate.main()
        return output.getvalue()

    def test_train_gpt_checkpoint_reconstructs_without_architecture_flags(self):
        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)

        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved["model_config"], self.config)
        # The 8-token sinusoidal table forces the CLI to read max_seq_len from the checkpoint.
        self.assertTrue(self.run_cli("--max_new_tokens", "12").startswith("ab"))

    def test_stop_at_eos_allows_and_stops_on_eos(self):
        eos = Tokenizer.SPECIAL_TOKENS["<EOS>"]
        with torch.no_grad():
            self.model.decoder.fully_connected.weight.zero_()
            self.model.decoder.fully_connected.bias.zero_()
            self.model.decoder.fully_connected.bias[eos] = 1.0
        save_checkpoint(self.model, str(self.checkpoint), self.tokenizer, self.config)

        with patch.object(GPT, "generate", autospec=True, side_effect=GPT.generate) as spy:
            self.assertEqual(self.run_cli("--stop_at_eos", "--max_new_tokens", "5"), "ab\n")
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["eos_token_id"], eos)
        self.assertNotIn(eos, kwargs["forbidden_token_ids"])
        # Without the flag EOS stays forbidden, so the full budget of non-special tokens is generated.
        self.assertNotEqual(self.run_cli("--length", "5"), "ab\n")


if __name__ == "__main__":
    unittest.main()
