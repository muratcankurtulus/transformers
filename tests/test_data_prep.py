import hashlib
import tempfile
import unittest
from pathlib import Path

from data_prep.prepare_fineweb import normalize_document, write_corpus


def document_for_split(evaluation):
    for index in range(10000):
        text = (f"Document {index}. " + "English prose with Original Case and complete sentences. " * 8).strip()
        digest = hashlib.sha256(text.encode()).digest()
        if (int.from_bytes(digest[:8], "big") % 100 == 0) == evaluation:
            return text
    raise AssertionError("No matching synthetic document")


class DataPrepTests(unittest.TestCase):
    def test_preserves_case_and_unicode_while_normalizing_line_endings(self):
        self.assertEqual(normalize_document("  Cafe\u0301\r\n\r\nOriginal CASE  \r\n"), "Café\nOriginal CASE")

    def test_splits_documents_before_sampling_and_never_uses_eval_for_tokenizer(self):
        training, evaluation = document_for_split(False), document_for_split(True)
        records = [{"text": training}, {"text": training}, {"text": evaluation}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            counts = write_corpus(records, output, 100000, 100000, 100000)
            self.assertEqual(counts["train_documents"], 1)
            self.assertEqual(counts["eval_documents"], 1)
            self.assertEqual((output / "train.txt").read_text(), training + "\n\n")
            self.assertEqual((output / "eval.txt").read_text(), evaluation + "\n\n")
            self.assertEqual((output / "tokenizer_sample.txt").read_text(), training + "\n\n")

    def test_bounded_output_retains_complete_documents_and_refuses_overwrite(self):
        training, evaluation = document_for_split(False), document_for_split(True)
        records = [{"text": training}, {"text": evaluation}, {"text": "Should never be consumed"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            counts = write_corpus(iter(records), output, 1, 1, 1)
            self.assertEqual(counts["train_bytes"], len(training.encode()) + 2)
            self.assertEqual(counts["eval_bytes"], len(evaluation.encode()) + 2)
            with self.assertRaises(FileExistsError):
                write_corpus(records, output, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
