import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from pretokenize import pretokenize_file_streaming
from tokenizer import Tokenizer
from train_gpt import Dataset, load_checkpoint_state, main, save_checkpoint


def train_tokenizer(text, vocab_size):
    tokenizer = Tokenizer(vocab_size=vocab_size)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tokenizer.train_streaming(iter([text]), show_progress=False)
    return tokenizer


class TokenizerIdsTests(unittest.TestCase):
    def test_all_bytes_and_specials_have_disjoint_defined_ids(self):
        tokenizer = Tokenizer(vocab_size=260)
        for byte in range(256):
            self.assertEqual(tokenizer.vocab[byte], bytes([byte]))
        self.assertEqual(set(tokenizer.SPECIAL_TOKENS.values()), {256, 257, 258, 259})
        self.assertEqual(set(tokenizer.vocab), set(range(tokenizer.vocab_size)))

    def test_control_characters_and_unicode_round_trip(self):
        tokenizer = Tokenizer(vocab_size=260)
        for text in ("", "A", "\x00A\x01B\x02C\x03D", "".join(map(chr, range(256))), "Türkçe 日本語 😀"):
            with self.subTest(text=text):
                tokens = tokenizer.encode(text)
                self.assertTrue(set(tokens).isdisjoint(tokenizer.SPECIAL_TOKENS.values()))
                self.assertEqual(tokenizer.decode(tokens), text)

    def test_explicit_specials_are_removed_but_control_bytes_and_literal_strings_are_preserved(self):
        tokenizer = Tokenizer(vocab_size=260)
        text = "\x00\x01\x02\x03 literal <PAD> <UNK> <BOS> <EOS>"
        tokens = tokenizer.add_special_tokens(tokenizer.encode(text))
        tokens.extend([tokenizer.SPECIAL_TOKENS["<PAD>"], tokenizer.SPECIAL_TOKENS["<UNK>"]])
        self.assertEqual(tokenizer.decode(tokens), text)
        self.assertEqual(tokenizer.decode(torch.tensor(tokens)), text)

    def test_merges_preserve_control_bytes(self):
        text = "\x00\x01\x02\x03 " * 8
        tokenizer = train_tokenizer(text, 264)
        self.assertTrue(tokenizer.merges)
        self.assertGreaterEqual(min(tokenizer.merges.values()), 260)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(set(tokenizer.vocab), set(range(tokenizer.vocab_size)))

    def test_vocabulary_cannot_omit_bytes_or_specials(self):
        for size in (0, 255, 256, 259):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "at least 260"):
                Tokenizer(vocab_size=size)

    def test_capacity_uses_actual_id_range_when_training_stops_early(self):
        tokenizer = train_tokenizer("aaaa", 1024)
        self.assertEqual(tokenizer.vocab_size, 262)
        self.assertEqual(set(tokenizer.vocab), set(range(tokenizer.vocab_size)))
        tokenizer.validate_vocab_size(262)
        for size in (261, 1024):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "actual vocabulary size: 262"):
                tokenizer.validate_vocab_size(size)

    def test_save_load_preserves_ids_size_and_identity(self):
        tokenizer = train_tokenizer("\x00\x01\x02\x03 aaaa Türkçe", 268)
        text = "\x00A\x01B\x02C\x03D aaaa Türkçe 😀"
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save(directory)
            loaded = Tokenizer.load(directory)
        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded.metadata, tokenizer.metadata)
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_legacy_tokenizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab = {idx: bytes([idx]) for idx in range(256)}
            vocab.update({idx: token.encode() for idx, token in enumerate(("<PAD>", "<UNK>", "<BOS>", "<EOS>"))})
            joblib.dump(vocab, Path(directory) / "vocab")
            joblib.dump({}, Path(directory) / "merges")
            with self.assertRaisesRegex(ValueError, "Legacy tokenizer.*train new checkpoints"):
                Tokenizer.load(directory)

    def test_overwritten_byte_is_rejected_even_with_metadata(self):
        tokenizer = Tokenizer(vocab_size=260)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save(directory)
            tokenizer.vocab[0] = b"<PAD>"
            joblib.dump(tokenizer.vocab, Path(directory) / "vocab")
            with self.assertRaisesRegex(ValueError, "byte/special IDs"):
                Tokenizer.load(directory)

    def test_merge_cannot_reuse_a_special_id(self):
        tokenizer = train_tokenizer("ab", 261)
        tokenizer.merges[(97, 98)] = 259
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "merge IDs"):
                tokenizer.save(directory)


class TokenizerArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.tokenizer = Tokenizer(vocab_size=260)
        self.tokenizer_path = self.directory / "tokenizer"
        self.tokenizer.save(self.tokenizer_path)
        self.text = "\x00A\x01B\x02C\x03D Türkçe 😀\nAnother line\n"
        self.input_path = self.directory / "input.txt"
        self.input_path.write_text(self.text, encoding="utf-8")

    def pretokenize(self, output_format):
        path = self.directory / f"tokens.{output_format}"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            pretokenize_file_streaming(
                str(self.tokenizer_path), str(self.input_path), str(path), output_format=output_format, chunk_size=1
            )
        return str(path)

    def test_regenerated_bin_and_pt_data_preserve_control_bytes(self):
        for output_format in ("bin", "pt"):
            with self.subTest(output_format=output_format):
                path = self.pretokenize(output_format)
                with redirect_stdout(io.StringIO()):
                    dataset = Dataset(path, 2, self.tokenizer, "default")
                self.assertEqual(dataset.data.tolist(), self.tokenizer.encode(self.text))
                self.assertEqual(self.tokenizer.decode(dataset.data.tolist()), self.text)
                with open(path + ".tokenizer.json", encoding="utf-8") as f:
                    self.assertEqual(json.load(f), self.tokenizer.metadata)

    def test_unversioned_data_is_rejected_in_both_formats(self):
        for output_format in ("bin", "pt"):
            with self.subTest(output_format=output_format):
                path = self.pretokenize(output_format)
                Path(path + ".tokenizer.json").unlink()
                with self.assertRaisesRegex(ValueError, "regenerate tokenized data"):
                    Dataset(path, 2, self.tokenizer, "default")

    def test_same_sized_different_tokenizer_data_is_rejected(self):
        tokenizer = train_tokenizer("ab", 261)
        other = train_tokenizer("ac", 261)
        self.assertEqual(tokenizer.vocab_size, other.vocab_size)
        path = self.pretokenize("bin")
        tokenizer.save_metadata(path + ".tokenizer.json")
        with self.assertRaisesRegex(ValueError, "Incompatible.*identity"):
            Dataset(path, 2, other, "default")

    def test_pretokenized_flag_cannot_bypass_identity_validation(self):
        path = self.pretokenize("pt")
        Path(path + ".tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "regenerate tokenized data"):
            main(
                tokenizer_path=str(self.tokenizer_path),
                train_data_path=path,
                eval_data_path=path,
                epochs=1,
                experiment_name="unused",
                tokenizer_type="default",
                eval_interval=100,
                use_pretokenized=True,
                weight_decay=0,
                early_stopping_patience=1,
                dropout_rate=0,
                embed_dim=8,
                tgt_vocab_size=260,
                seq_len=2,
                num_layers=1,
                expansion_factor=4,
                n_heads=2,
                batch_size=1,
                shuffle=False,
            )

    def test_pretokenized_data_requires_a_custom_tokenizer(self):
        path = self.pretokenize("bin")
        with self.assertRaisesRegex(ValueError, "tokenizer is required"):
            Dataset(path, 2, None, "default")

    def test_control_and_special_ids_train_and_checkpoint_on_cpu(self):
        model = nn.Sequential(nn.Embedding(260, 8), nn.Linear(8, 260))
        tokens = torch.tensor(self.tokenizer.add_special_tokens(self.tokenizer.encode("\x00A\x01B\x02C\x03D")))
        src, tgt = tokens[:-1], tokens[1:]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = nn.CrossEntropyLoss()(model(src), tgt)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        optimizer.step()
        path = str(self.directory / "model.pth")
        save_checkpoint(model, path, self.tokenizer)
        restored = nn.Sequential(nn.Embedding(260, 8), nn.Linear(8, 260))
        restored.load_state_dict(load_checkpoint_state(path, self.tokenizer))
        torch.testing.assert_close(restored(src), model(src))

    def test_legacy_and_wrong_version_checkpoints_are_rejected(self):
        model = nn.Linear(8, 260)
        path = str(self.directory / "model.pth")
        torch.save(model.state_dict(), path)
        with self.assertRaisesRegex(ValueError, "train new checkpoints"):
            load_checkpoint_state(path, self.tokenizer)
        metadata = {**self.tokenizer.metadata, "format_version": 1}
        torch.save({"model_state_dict": model.state_dict(), "tokenizer": metadata}, path)
        with self.assertRaisesRegex(ValueError, "Incompatible.*identity"):
            load_checkpoint_state(path, self.tokenizer)

    def test_same_sized_different_tokenizer_checkpoint_is_rejected(self):
        tokenizer = train_tokenizer("ab", 261)
        other = train_tokenizer("ac", 261)
        path = str(self.directory / "model.pth")
        save_checkpoint(nn.Linear(8, 261), path, tokenizer)
        with self.assertRaisesRegex(ValueError, "Incompatible.*identity"):
            load_checkpoint_state(path, other)

    def test_tiktoken_artifact_formats_are_unchanged(self):
        path = self.directory / "tiktoken.bin"
        tokens = np.array([10, 20, 30], dtype=np.uint16)
        path.write_bytes(b"BPE1" + struct.pack("<BQ", 2, len(tokens)) + tokens.tobytes())
        with redirect_stdout(io.StringIO()):
            dataset = Dataset(str(path), 2, None, "tiktoken")
        self.assertEqual(dataset.data.tolist(), tokens.tolist())
        model = nn.Linear(8, 32)
        checkpoint_path = str(self.directory / "tiktoken.pth")
        save_checkpoint(model, checkpoint_path, None)
        state = load_checkpoint_state(checkpoint_path, None)
        self.assertEqual(set(state), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
