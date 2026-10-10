# A complete RTX 2070 Super training cycle

This workflow targets an RTX 2070 Super with 8 GB VRAM and 24–48 hours of GPU training.
It trains this repository's GPT from random initialization on a bounded FineWeb-Edu subset.
The goal is a small English completion model. Useful general chat requires a much stronger
pretrained model; instruction fine-tuning cannot manufacture knowledge missing from pretraining.

## Dataset choice

Use [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for this experiment:
it supplies filtered English educational web text without maintaining a raw Common Crawl cleaning pipeline.
Use [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) instead if coherent simple stories
are more important than broad English coverage. For useful chat on this hardware, start with a pretrained
small model such as [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
then adapt it to a specific task with LoRA and an appropriate instruction dataset. That is a separate
Hugging Face training workflow and does not train this repo's architecture from scratch.
Retain that model's pretrained tokenizer. Training a replacement BPE tokenizer would invalidate
the token-ID meaning learned by the pretrained weights.

The current `data_prep/download_cc_data.sh` expects a separately supplied `wet.paths` file, downloads one file
into the current directory, and does not populate the `wet/` directory expected by `exp_cc.py`.
The extractor hardcodes dated filenames and treats WARC headers as document content. It lowercases
all text and needs a separate FastText model. Those scripts need repair before they are a reliable
corpus pipeline. The new preparer preserves case and splits whole documents before tokenizer training.

## 1. Install on the GPU machine

Use Python 3.11 and Bash. Run from this worktree, or from a checkout of `feat/gpu-training-cycle`.
The commands below use a pinned CUDA 12.1 PyTorch wheel that supports Turing. An NVIDIA driver
compatible with CUDA 12.1 must already be installed. No external FlashAttention package is needed.

```bash
python3.11 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[data]'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
```

FP16 is the suitable mixed precision setting here. Do not select BF16 or install FlashAttention-2
for a 2070 Super. PyTorch scaled dot-product attention selects an available native backend.

## 2. Prepare the text corpus

```bash
python data_prep/prepare_fineweb.py \
  --output_dir toy_data/fineweb \
  --train_bytes 1000000000 \
  --eval_bytes 4000000 \
  --tokenizer_bytes 3000000
```

This streams the public `sample-10BT` configuration, shuffles with a fixed buffer/seed, and writes
approximately 1 GB of training text, 4 MB of held-out text, and a 3 MB tokenizer sample taken only
from training documents. Limits can exceed by one complete document. It records the actual dataset
commit and split rules in `manifest.json`. Identical normalized documents are deduplicated and never
cross splits; near-duplicate documents are not detected. Output files are never overwritten.
The full 10-billion-token dataset is not downloaded, although source shard/buffer downloads can
be larger than the selected text. Reserve several GB of disk space for the corpus, binaries, and cache.

Text normalization and tokenization use the CPU; their runtime is additional to the GPU training budget.
The custom Python BPE trainer rescans its word table on each merge, so this step may be slow.

## 3. Train the tokenizer and encode both splits

```bash
python src/tokenization.py \
  --train toy_data/fineweb/tokenizer_sample.txt \
  --output_dir toy_data/fineweb/tokenizer \
  --vocab_size 8192

python src/pretokenize.py file toy_data/fineweb/train.txt \
  --tokenizer toy_data/fineweb/tokenizer --format bin

python src/pretokenize.py file toy_data/fineweb/eval.txt \
  --tokenizer toy_data/fineweb/tokenizer --format bin

python src/pretokenize.py info toy_data/fineweb/train.bin
```

Keep `train.bin.tokenizer.json` and `eval.bin.tokenizer.json` beside their binaries. The trainer
reads the actual vocabulary size and checks tokenizer identity automatically.

## 4. Benchmark before committing the training budget

```bash
python src/train_budget.py \
  --tokenizer toy_data/fineweb/tokenizer \
  --train_data toy_data/fineweb/train.bin \
  --eval_data toy_data/fineweb/eval.bin \
  --output_dir runs/fineweb_2070_benchmark \
  --precision fp16 \
  --batch_size 8 --gradient_accumulation 4 \
  --max_steps 100
```

The model defaults are 8 layers, width 384, 6 attention heads, context 256, expansion 4, and dropout 0.1.
With an actual vocabulary of 8192, it has **20,486,912 parameters**. A batch of 8 with four accumulation
passes processes 8192 training tokens per optimizer step. The batch is a starting point, not a measured
VRAM guarantee. Inspect `tokens_per_second` and `peak_vram_gb` after the first 20 steps.

If the benchmark runs out of VRAM, use batch 4 / accumulation 8 in a **new output directory**.
If there is substantial headroom, benchmark batch 16 / accumulation 2 in another directory.
Use the fastest fitting setting consistently for the full run and resume.

Measured throughput sets the possible training budget:

| Measured tokens/s | Exposures in 24 hours | Exposures in 48 hours |
| ----------------- | --------------------: | --------------------: |
| 500               |          43.2 million |          86.4 million |
| 1000              |          86.4 million |         172.8 million |
| 2000              |         172.8 million |         345.6 million |

These are arithmetic upper estimates before evaluation/checkpoint overhead, not GPU benchmark claims.
The trainer randomly samples packed blocks with replacement. `tokens_seen` measures exposures,
including repeated blocks, rather than distinct corpus tokens. A fixed token budget is more informative
than the previous loader's heavily overlapping epochs.

## 5. Train for 36 hours

```bash
python src/train_budget.py \
  --tokenizer toy_data/fineweb/tokenizer \
  --train_data toy_data/fineweb/train.bin \
  --eval_data toy_data/fineweb/eval.bin \
  --output_dir runs/fineweb_2070 \
  --precision fp16 \
  --batch_size 8 --gradient_accumulation 4 \
  --max_hours 36
```

Use `--max_hours 24` or `48` to change the total budget. The trainer warms up for 200 optimizer steps,
then applies wall-clock cosine decay from a peak learning rate of 3e-4 to 3e-5. It clips gradients,
evaluates on a fixed bounded held-out sample every 200 steps, and saves optimizer/scaler/RNG state
in `last.pth` at least every 10 minutes. `best.pth` selects the lowest held-out loss. Logs are in
`metrics.jsonl`. The time budget is checked between optimizer steps; final evaluation/save takes extra time.
Abrupt process termination can lose work since the last save. Ctrl-C saves before exiting.

Resume an interrupted run with the same settings:

```bash
python src/train_budget.py \
  --tokenizer toy_data/fineweb/tokenizer \
  --train_data toy_data/fineweb/train.bin \
  --eval_data toy_data/fineweb/eval.bin \
  --output_dir runs/fineweb_2070 \
  --resume runs/fineweb_2070/last.pth \
  --precision fp16 \
  --batch_size 8 --gradient_accumulation 4 \
  --max_hours 36
```

The 36-hour budget includes previously saved active training time. Resume rejects different data,
tokenizer, model, batch, precision, optimizer, or seed settings. Use `last.pth` to resume and `best.pth`
for inference. Extending `--max_hours` also changes the remaining cosine learning-rate schedule.

## 6. Generate text and judge the result

```bash
python src/generate.py \
  --model_path runs/fineweb_2070/best.pth \
  --tokenizer_path toy_data/fineweb/tokenizer \
  --prompt "The water cycle is important because" \
  --length 150 --temperature 0.8 --top_k 50 --top_p 0.9
```

The new checkpoint supplies its architecture and context settings. Generation crops the context
to the trained length and supports sampling. Try several fixed prompts and seeds, for example
`--seed 42` and `--seed 43`, and compare early checkpoints with the selected best checkpoint.

Assess whether held-out loss decreases, whether short completions become grammatical, and whether
repetition declines across multiple prompts. A small model may still produce inaccurate claims,
weak long-range consistency, and poor answers to questions. Pretraining here learns continuation;
it does not add a chat template or supervised instruction training. There is no guarantee that a
20-million-parameter from-scratch run becomes a useful assistant in two days.

## Verification commands

```bash
PYTHONPATH=src:. python -m unittest discover -s tests
```

The new tests cover deterministic document splitting, train-only tokenizer sampling, packed next-token
targets, causal isolation, FP16 RoPE dtype, bounded generation, and checkpoint resume. CUDA throughput
and memory must be validated with the benchmark on the 2070 Super.
