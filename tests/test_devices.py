import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import generate
from devices import resolve_device
from gpt import GPT
from tokenizer import Tokenizer

SRC = Path(__file__).resolve().parents[1] / "src"
CORE_MODULES = (
    "blocks",
    "gpt",
    "transformer",
    "tokenizer",
    "tokenization",
    "pretokenize",
    "devices",
    "train_budget",
    "train_gpt",
    "generate",
    "benchmark",
)

# Any GPU query during import raises; then warning filters and compiler settings must be untouched.
IMPORT_PROBE = """
import importlib, sys, warnings
import torch

def probe(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"GPU probed at import time via {name}")

    return fail  # one function per name: torch._dynamo rejects a shared object under several names

for name in ("is_available", "device_count", "init", "current_device", "get_device_name", "set_device"):
    setattr(torch.cuda, name, probe(f"torch.cuda.{name}"))
torch.backends.mps.is_available = probe("torch.backends.mps.is_available")
filters = list(warnings.filters)
for module in sys.argv[1:]:
    importlib.import_module(module)
    assert warnings.filters == filters, f"{module} changed global warning filters"
    dynamo = sys.modules.get("torch._dynamo")
    assert dynamo is None or not dynamo.config.suppress_errors, f"{module} changed torch._dynamo config"
    assert not torch.cuda.is_initialized(), f"{module} initialized CUDA"
print("clean")
"""


class FakeCuda:
    """Patch CUDA discovery so device selection is testable on CPU-only machines."""

    def __init__(self, count):
        self.count = count

    def __enter__(self):
        self.patches = [
            patch("torch.cuda.is_available", return_value=self.count > 0),
            patch("torch.cuda.device_count", return_value=self.count),
            patch("torch.cuda.set_device"),
        ]
        mocks = [p.start() for p in self.patches]
        return mocks[2]

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_never_queries_cuda(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("probed")):
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_explicit_non_default_index_is_selected_and_made_current(self):
        with FakeCuda(4) as set_device, patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            self.assertEqual(resolve_device("cuda:2"), torch.device("cuda", 2))
        set_device.assert_called_once_with(2)

    def test_bare_cuda_follows_local_rank_then_defaults_to_zero(self):
        with FakeCuda(4) as set_device, patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda", 3))
        set_device.assert_called_once_with(3)
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        with FakeCuda(1), patch.dict(os.environ, env, clear=True):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda", 0))

    def test_rejects_missing_out_of_range_and_unsupported_devices(self):
        with FakeCuda(0), self.assertRaisesRegex(RuntimeError, "unavailable"):
            resolve_device("cuda")
        with FakeCuda(2), self.assertRaisesRegex(ValueError, "only 2 visible"):
            resolve_device("cuda:2")
        with FakeCuda(2), patch.dict(os.environ, {"LOCAL_RANK": "5"}), self.assertRaises(ValueError):
            resolve_device("cuda")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_device("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_cuda_selection_moves_model_to_selected_gpu(self):
        index = torch.cuda.device_count() - 1
        device = resolve_device(f"cuda:{index}")
        model = GPT(16, 8, num_layers=1, n_heads=2, dropout_rate=0.0).to(device)
        output = model.generate([1, 2, 3], 2)
        self.assertEqual(output.device, device)
        self.assertEqual(torch.cuda.current_device(), index)


class ImportPolicyTests(unittest.TestCase):
    def test_core_modules_import_without_gpu_probing_or_global_side_effects(self):
        result = subprocess.run(
            [sys.executable, "-c", IMPORT_PROBE, *CORE_MODULES],
            cwd=SRC,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "clean")


class CpuGenerationCliTests(unittest.TestCase):
    def test_generate_cli_runs_on_cpu_from_a_saved_checkpoint(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tokenizer = Tokenizer(vocab_size=260)
            tokenizer.save(directory / "tokenizer")
            config = {"tgt_vocab_size": 260, "embed_dim": 16, "num_layers": 1, "n_heads": 2}
            config |= {"expansion_factor": 4, "dropout_rate": 0.0, "pos_encoding_type": "rotary", "max_seq_len": 8}
            model = GPT(**config)
            checkpoint = directory / "model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "tokenizer": tokenizer.metadata,
                    "seq_len": 8,
                },
                checkpoint,
            )
            argv = ["generate.py", "--prompt", "ab", "--model_path", str(checkpoint)]
            argv += ["--tokenizer_path", str(directory / "tokenizer"), "--length", "3", "--device", "cpu"]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                generate.main()
        self.assertTrue(output.getvalue().startswith("ab"))


if __name__ == "__main__":
    unittest.main()
