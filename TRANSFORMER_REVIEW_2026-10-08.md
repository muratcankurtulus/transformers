**Transformer repository review — 8 October 2026**

Reviewed revision: `f58c1d9`, together with the current working tree. This review covers all source files, the README, dependency metadata, and formatting configuration. The implementation findings refer to your code; recommendations about current practice use the linked primary papers and official documentation.

**Overall assessment**

This is a useful learning implementation with recognizable transformer components. The GPT path has the essentials: token embeddings, multi-head attention, attention scaling, a causal mask, residual connections, normalization, a feed-forward network, and a vocabulary prediction head. The training data correctly pairs each input token with the following token. You also use `ModuleList`, registered buffers, `AdamW`, `zero_grad(set_to_none=True)`, evaluation mode, pinned memory, and non-blocking transfers appropriately in several places.

However, I would not yet use this version as a trustworthy baseline for architecture comparisons or expensive training. Some problems change the model's behavior or prevent execution. Other problems spend much more compute than the dataset size suggests. Distributed training is not implemented.

The first goal should be a correct, reproducible small model. After that, improve throughput and introduce modern architecture choices through measured comparisons. Adding every feature from a frontier model would make this project harder to understand before its foundations are reliable.

**What I verified, and what remains unmeasured**

I ran small synthetic checks in an isolated temporary environment using Python 3.13.1 and PyTorch 2.14.0 on CPU. The repository pins PyTorch 2.4.1, so these are checks against a current framework version, not a recreation of your original CUDA environment. All source files passed Python syntax parsing. A wheel built successfully from a temporary copy of the source and metadata.

The tiny GPT forward pass produced finite outputs with shape `[1, 4, 32]`. Changing future tokens left earlier logits unchanged when I supplied a correct CPU causal mask. That supports the basic GPT attention logic. Its built-in mask helper itself failed on CPU because it forces CUDA.

Other checks reproduced the tokenizer round-trip and decoding problems, incorrect sinusoidal frequencies, BF16 RoPE failure, cross-attention failure with unequal sequence lengths, float-mask failure, encoder–decoder softmax failure, and context overflow described below. The RoPE cache device problem was also reproduced using PyTorch's meta device; that is device-placement evidence, not a CUDA benchmark.

There is no CUDA available in this review environment. I did not run full training, measure GPU utilization, validate FlashAttention dispatch, benchmark compilation, test multiple GPUs, inspect your training corpus, or load existing tokenizer/checkpoint artifacts. Statements about throughput improvements are engineering expectations unless a measurement is explicitly given. Model quality and convergence cannot be judged from source code alone.

**Priority order**

| Priority                        | Work                                                                                                                      | Why it comes first                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| P0 — correctness                | Repair token IDs; encoder–decoder masks, cross-attention, and output logits; position encodings; checkpoint configuration | These can corrupt data, produce wrong predictions, or prevent execution.    |
| P1 — useful single-GPU training | Fix dataset sampling, enable correct SDPA and mixed precision, improve evaluation and restart support                     | These determine whether compute produces useful, reproducible results.      |
| P2 — scalable experiments       | Add DDP, distributed data sampling, global metrics, and coordinated checkpoints                                           | Multiple processes otherwise train separate models or duplicate work.       |
| P3 — architecture experiments   | Compare pre-normalization, RMSNorm, SwiGLU, GQA, and longer contexts                                                      | These are meaningful only after the baseline is trustworthy.                |
| P4 — advanced research          | Consider MoE, hybrid/sparse attention, FP8, multi-token prediction, and advanced parallelism                              | Their costs and benefits depend heavily on model scale, hardware, and task. |

**1. Transformer architecture: correctness problems**

**1.1 Encoder–decoder cross-attention swaps the query and key — P0.**

Where: [the attention call in `TransformerEncoderBlock`](/Users/maskedpirate/repos/transformers/src/blocks.py:278), reached from [the decoder's cross-attention path](/Users/maskedpirate/repos/transformers/src/blocks.py:336).

What is lacking: `MultiHeadAttention.forward` expects `(query, key, value)`, but the block calls it as `(key, query, value)`. For decoder cross-attention, the decoder should ask the questions; the encoder should supply the keys and values to look up.

Why it matters: with five source tokens and three target tokens, my check failed because the attention matrix and value tensor had incompatible lengths. Equal lengths hide that shape problem while still computing the wrong operation.

How to fix it: use explicit keyword arguments: `self.attention(query=query, key=key, value=value)`. Prefer a dedicated cross-attention sublayer instead of reusing a class named `TransformerEncoderBlock`. Verify that output length follows the target/query length, and test source and target lengths that differ. This is a basic encoder–decoder requirement explained in [Attention Is All You Need](https://arxiv.org/html/1706.03762v7).

**1.2 The encoder–decoder causal mask has the wrong type and blocks the diagonal — P0.**

Where: [`Transformer.make_tgt_mask`](/Users/maskedpirate/repos/transformers/src/transformer.py:19) and [the attention mask application](/Users/maskedpirate/repos/transformers/src/blocks.py:177).

What is lacking: the mask is a floating-point upper triangle, while `masked_fill_` expects a boolean mask. Its upper triangle also starts on the diagonal.

Why it matters: the current operation raises a runtime error. Converting it to boolean without changing the diagonal would still be wrong: each token must see itself and earlier tokens. The first row would have every key blocked, causing an all-negative-infinity softmax and potentially NaNs.

How to fix it: for the current attention implementation, create a boolean upper triangle with `diagonal=1` on the input device. Add separate padding masks where needed. Test a one-token sequence and verify that future-token changes cannot affect earlier outputs. The GPT helper uses the correct diagonal, but also needs its hard-coded CUDA device removed.

**1.3 The encoder–decoder output softmax uses the batch axis — P0.**

Where: [`TransformerDecoder.forward`](/Users/maskedpirate/repos/transformers/src/blocks.py:370).

What is lacking: `F.softmax(self.fully_connected(x))` omits `dim`. For the three-dimensional output in my PyTorch check, the implicit axis was the batch axis.

Why it matters: with batch size one and vocabulary size 32, every output became exactly `1.0`; each token's vocabulary scores summed to 32. These are not useful vocabulary probabilities. Passing these probabilities to cross-entropy would introduce another error because cross-entropy expects raw logits.

How to fix it: return `self.fully_connected(x)` directly, as your GPT decoder already does. Apply `softmax(dim=-1)` only when probabilities are needed for sampling or display. This keeps the training interface consistent across both model families.

**1.4 The sinusoidal position formula is incorrect — P0.**

Where: [`PositionalEncoding.__init__`](/Users/maskedpirate/repos/transformers/src/blocks.py:103).

What is lacking: sine and cosine in each pair use different frequencies, and the exponent is doubled relative to the already-even dimension index. The standard pair uses `sin(position / 10000^(2j/d))` and `cos(position / 10000^(2j/d))` for dimensions `2j` and `2j+1`.

Why it matters: this no longer implements the encoding named in the comments. For width four and position one, the expected first cosine is about `0.5403`; the implementation produces about `0.99995`. The next sine is about `0.0001` instead of `0.0100`.

How to fix it: compute frequencies with a vectorized `arange(0, embed_dim, 2) / embed_dim`, then assign the same frequency to each sine/cosine pair. Validate supported dimensions. Vectorization also removes the nested Python initialization loops. Compare a few small examples against the positional encoding definition in [the original paper](https://arxiv.org/html/1706.03762v7).

**1.5 Selecting RoPE still adds sinusoidal positions — P0/P1.**

Where: [GPT decoder construction and forward](/Users/maskedpirate/repos/transformers/src/blocks.py:229), and [encoder construction](/Users/maskedpirate/repos/transformers/src/blocks.py:306).

What is lacking: the embedding path always adds sinusoidal positions, even when the attention path applies RoPE. Some encoder–decoder constructor choices also fall back to default positional/dropout settings in nested blocks.

Why it matters: a run labeled “rotary” is actually a combination of two positional methods. That combination is not inherently impossible, but it is not the controlled choice the API advertises. It also leaves an unnecessary fixed-length buffer limiting the rotary model.

How to fix it: make the options explicit. A standard rotary path rotates queries and keys and does not add this absolute sinusoidal encoding. A sinusoidal path adds the embedding and does not rotate Q/K. Propagate configuration consistently and test both paths. Read [RoFormer](https://arxiv.org/abs/2104.09864) for the underlying idea.

**1.6 RoPE is not ready for mixed precision, device changes, or cached generation — P1.**

Where: [`RotaryPositionalEncoding`](/Users/maskedpirate/repos/transformers/src/blocks.py:28).

What is lacking: `view_as_complex` rejects BF16 tensors in the check. FP16 input is promoted to FP32 by multiplication with the FP32 rotation. Cache extension creates positions on CPU, even when `inv_freq` has moved to another device. Every call also starts positions at zero.

Why it matters: simply enabling BF16 training will fail. Enabling FP16 can produce inconsistent attention dtypes. Longer sequences can fail during cache extension on GPU. Future KV caching would rotate each new token at position zero unless you add its true position.

How to fix it: implement the pairwise rotation with real tensors. Calculate frequencies in FP32 on the correct device, then apply the rotation with a deliberate dtype policy and return the intended attention dtype. Accept position IDs or a cache offset. Require an even rotary dimension. Treat cosine/sine tables as reconstructible caches, generally using non-persistent buffers, rather than essential checkpoint state.

**1.7 Padding and document boundaries are not represented — P1, depending on the task.**

Where: [attention interfaces](/Users/maskedpirate/repos/transformers/src/blocks.py:159), [dataset construction](/Users/maskedpirate/repos/transformers/src/train_gpt.py:28).

What is lacking: there are no source padding masks, target padding masks, or explicit document boundaries. Defining PAD/BOS/EOS tokens does not help unless the dataset and loss actually use them.

Why it matters: the current fixed-length continuous GPT stream does not need padding, so absence of padding is not a bug in that narrow path. It becomes a correctness problem for variable-length translation batches. Joining unrelated documents without an EOS separator also trains the model to predict artificial transitions.

How to fix it: choose a document policy and insert EOS at genuine boundaries. For padded examples, block padded keys and exclude padded labels from the loss using an appropriate `ignore_index`. If packed examples must stay independent, add attention boundaries and explicit position handling; an EOS token alone does not isolate their attention.

**1.8 Generation and checkpoint loading are incomplete — P0/P1.**

Where: [`generate.py`](/Users/maskedpirate/repos/transformers/src/generate.py:18), [`GPT.generate`](/Users/maskedpirate/repos/transformers/src/gpt.py:57), [`Transformer.generate`](/Users/maskedpirate/repos/transformers/src/transformer.py:24).

What is lacking: generation hard-codes a 160-wide, three-layer GPT, while training defaults to width 384 and six layers. `--vocab_size` is required but unused. Checkpoints contain no architecture or tokenizer identity. GPT generation never stops at EOS and repeatedly processes the entire prefix. It fails when the prefix exceeds the fixed positional table. The encoder–decoder generator discards its history, reuses its original mask, bases generation length on source length, and assumes scalar batch output.

Why it matters: a normal training checkpoint will not load into the generation defaults. Generation may crash after enough tokens. The encoder–decoder loop is not implementing normal autoregressive decoding.

How to fix it: reconstruct the architecture from checkpoint metadata, verify the tokenizer fingerprint, and load onto an explicit device. Specify `max_new_tokens`, validate prompt shape and length, append generated tokens to the history, and stop on EOS. Define a context-limit policy. Add a per-layer KV cache so generation reuses earlier keys and values, with correct position offsets and causal alignment. Cropping context is a policy decision and should not silently replace longer-context training. Greedy decoding is a valid baseline; temperature/top-p can be added as explicit options.

**2. What is lacking compared with modern transformer architectures**

There is no single “October 2026 SOTA architecture.” Different leading systems optimize different tasks and hardware. Public examples already show substantial variation: Qwen3 documents a dense baseline with pre-normalization, RMSNorm, SwiGLU, GQA, RoPE, and QK normalization; Qwen3.8-27B documents a hybrid stack of recurrent/linear-attention and full-attention layers; DeepSeek-V4 documents compressed attention, MoE, modified residual connections, and a different optimizer. These are examples of the frontier, not evidence that every feature must be added to this repository. [Qwen3 architecture](https://arxiv.org/html/2505.09388v1#S2), [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348).

**2.1 Normalization happens after the residual addition.**

Where: [`GPTDecoderBlock.forward`](/Users/maskedpirate/repos/transformers/src/blocks.py:207).

Your block uses post-normalization: `norm(x + attention(x))`. It is a legitimate architecture, especially in a small educational model. It is less attractive as a default for scaling depth because optimization can be more sensitive. Pre-normalization preserves a more direct residual path: `x = x + attention(norm(x))`, followed by `x = x + ffn(norm(x))`.

Fix: introduce a pre-normalization option and compare it at equal compute and data. RMSNorm is a useful accompanying experiment: it rescales activations without subtracting their mean. Keep the final normalization. This changes model behavior and requires a fresh training comparison. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).

**2.2 The GPT feed-forward network uses plain ReLU.**

Where: [GPT feed-forward layers](/Users/maskedpirate/repos/transformers/src/blocks.py:200).

ReLU works, but a gated network such as SwiGLU gives the model a learned way to control which transformed features pass through. It is a common modern choice. A typical form is `down(silu(gate(x)) * up(x))`.

Fix: implement it as a configurable alternative. It has three large projections rather than two, so keeping the same intermediate width makes the comparison larger and more expensive. Approximately `8/3 * embed_dim` gives a similar large-matrix parameter budget to a conventional `4 * embed_dim` feed-forward network; round to a hardware-friendly size and count the actual parameters. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

**2.3 Every query head has its own key/value head.**

Where: [attention projections](/Users/maskedpirate/repos/transformers/src/blocks.py:154).

This standard multi-head design is correct. Grouped-query attention lets several query heads share fewer key/value heads, reducing their projection size and KV-cache memory. Its clearest benefit here would appear after cached generation is implemented; it does not automatically produce a proportional training speedup.

Fix: separate `n_query_heads` from `n_kv_heads`, enforce compatible divisibility, and compare quality and cache memory. Do not simply shrink K/V without updating shapes and the attention backend. [GQA paper](https://arxiv.org/abs/2305.13245).

**2.4 Long context is not a complete capability.**

The default context is only 256 tokens, and the current positional embedding table imposes a hard limit. Making RoPE's cache longer only makes execution possible; it does not prove that the model can use positions or dependencies it never learned.

Fix: first repair position handling and generation, then train and evaluate at longer contexts. Measure memory, throughput, and tasks that actually require distant information. Consider RoPE scaling and longer-context adaptation only for a defined target. Efficient attention kernels help memory usage, but ordinary dense attention still performs quadratic attention work.

**2.5 Other architecture choices need explicit experiments.**

Your input embedding and output head are separate. Weight tying can save parameters and is a useful small-model experiment, especially with a larger vocabulary, but it is not mandatory in every modern LLM. Q/K normalization can be tested for attention stability. Initialization currently uses PyTorch defaults; a documented initialization policy, including residual projection scaling when increasing depth, would make comparisons more controlled.

Dropout is largely hard-coded to 0.2 and not exposed by the training configuration. `GPTDecoder.embed_dropout` is constructed but never called. Expose the rate and either apply or remove that unused module. Large-data pretraining and small-data learning experiments can require very different regularization, so avoid choosing a rate solely by copying a large model.

**2.6 Frontier-scale features are optional research directions.**

MoE routes tokens through a subset of many feed-forward experts. It adds capacity without activating every parameter, but introduces routing, balancing, expert communication, and capacity management. Hybrid or compressed attention can make very long contexts cheaper, with their own quality and kernel tradeoffs. FP8 requires hardware support and carefully managed numerical scaling. Multi-token prediction changes the training objective and can support other decoding techniques. Modified residual connections and Muon are further research directions documented by DeepSeek-V4. Read these after establishing a strong dense baseline. [DeepSeek-V4 official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro).

The larger capability gap also includes diverse, well-curated data, deduplication, a suitable token budget, robust evaluation, and task-specific post-training. Architecture alone will not turn a small next-token training experiment into a frontier assistant. Qwen3.8's official card explicitly distinguishes pretraining and post-training, illustrating that broader scope. [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B).

**3. Training performance and training quality**

**3.1 Stride-one windows make an epoch far more expensive than it appears — P1.**

Where: [`Dataset.valid_indices` and slices](/Users/maskedpirate/repos/transformers/src/train_gpt.py:38).

What is lacking: every valid token position becomes a starting point. For `N` tokens and sequence length `S`, there are `N-S` examples, each with `S` targets. An epoch therefore processes `(N-S)*S` target positions, not approximately `N`.

Why it matters: at the default `S=256`, an interior token is trained on about 256 times per epoch, at different positions and with different context lengths. Those contexts are not identical, so this is not automatically an invalid sampling scheme. But it is very expensive as the default definition of an epoch, and it makes “100 epochs” misleading. My ten-token, length-four example produced six windows and 24 target positions.

How to fix it: use contiguous packed blocks with stride about `S`, or randomly sample starting positions for a fixed number of optimizer steps. Preserve the next-token overlap needed for targets. Track total processed tokens and unique corpus tokens separately. Define training by a token/step budget. For sliding-window evaluation, score each intended target once or explicitly document repeated scoring; otherwise interior tokens are weighted many times.

**3.2 FlashAttention is installed but not used — P1.**

Where: [manual attention](/Users/maskedpirate/repos/transformers/src/blocks.py:177), [dependency declaration](/Users/maskedpirate/repos/transformers/pyproject.toml:10).

What is lacking: the code explicitly creates attention scores and weights with dimensions `[batch, heads, sequence, sequence]`. There is no FlashAttention call and no SDPA call. Installing a package does not replace those operations.

Why it matters: one FP32 attention matrix at the default batch 64, six heads, and sequence 256 occupies about 96 MiB. At sequence 2048 it is about 6 GiB. That is one matrix in one layer, before other activations and gradients.

How to fix it: use PyTorch's `scaled_dot_product_attention` and verify the selected kernel on your CUDA hardware. For ordinary square GPT training, use `is_causal=True`. With SDPA boolean masks, `True` means allowed; your current mask uses `True` for blocked. Invert it when migrating. Pass zero attention dropout during evaluation. Backend availability depends on dtype, shape, device, and masking. FlashAttention avoids storing the full attention matrix; it does not make dense attention computation linear. [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

**3.3 Training uses full precision without an explicit precision policy — P1.**

Where: [training forward/backward](/Users/maskedpirate/repos/transformers/src/train_gpt.py:123).

Why it is lacking: compatible accelerators can execute many operations faster and with less activation memory in BF16 or FP16. Your loop does not use autocast, and RoPE currently prevents a straightforward BF16 conversion.

Fix: repair RoPE first, then use `torch.autocast` with a supported dtype. BF16 is a sensible first CUDA option when supported; FP16 commonly needs `torch.amp.GradScaler`. Keep deliberately sensitive calculations in FP32. For FP16, unscale gradients before clipping. Compare loss and gradients against the FP32 baseline. FP8 is a later, hardware-dependent experiment. [Current AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html).

**3.4 Masks, logging, and data loading add avoidable overhead — P1/P2.**

Where: [mask allocation](/Users/maskedpirate/repos/transformers/src/gpt.py:53), [loader setup](/Users/maskedpirate/repos/transformers/src/train_gpt.py:100), [per-step logging](/Users/maskedpirate/repos/transformers/src/train_gpt.py:131).

You allocate a dense mask on CPU and transfer it every step, even though training shapes are fixed. You also call `loss.item()` repeatedly and update progress text each batch; obtaining a CUDA scalar introduces synchronization. Four DataLoader workers are fixed even though samples are already cheap tensor slices.

Fix: use implicit SDPA causality or reuse an appropriate device mask. Log at a controlled interval, retaining loss sums on device between reports where practical. Benchmark worker counts including zero; more workers are not automatically faster. If workers help, consider persistent workers and measured prefetch settings. Your existing pinned memory and non-blocking copies are useful, but they do not prove that transfers overlap efficiently. [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

**3.5 Compilation and optimizer performance are unused opportunities — P2.**

`torch.compile` is commented out. Separate Q/K/V projections and many small operations can add launch overhead. A fused QKV projection is useful for self-attention, and supported fused/foreach AdamW implementations may reduce optimizer overhead.

Fix: profile first, then compare eager and compiled execution with the same shapes and precision. Exclude compilation warmup from steady-state timing but report its startup cost. Correct the cache/device behavior before relying on compilation. PyTorch 2.14 adds experimental complex-tensor compilation support, so complex arithmetic should not be described as universally unsupported; real-valued RoPE still simplifies the BF16 problem. [PyTorch 2.14 release notes overview](https://pytorch.org/blog/pytorch-2-14-release-blog/).

**3.6 Optimization and regularization need controls — P1/P2.**

Where: [AdamW setup](/Users/maskedpirate/repos/transformers/src/train_gpt.py:90).

The learning rate stays at `1e-4`; there is no warmup, decay schedule, gradient accumulation, clipping, or finite-loss handling. Weight decay is AdamW's default and applies to all parameters in one group. These choices can work for a small run, but they provide little control over convergence or scaling.

Fix: expose learning rate, betas, weight decay, dropout, and a step/token budget. Test warmup plus a documented decay schedule. Consider excluding normalization parameters and biases from weight decay as an explicit experiment. Add gradient accumulation for a controllable effective batch and gradient clipping for a defined stability policy. Detect non-finite loss/gradients and report the failing configuration. Do not assume one learning rate, optimizer, or clipping threshold is optimal for all model sizes.

**3.7 Evaluation is costly and its aggregate is slightly biased — P1.**

Where: [`evaluate`](/Users/maskedpirate/repos/transformers/src/train_gpt.py:53), [evaluation scheduling](/Users/maskedpirate/repos/transformers/src/train_gpt.py:133).

You evaluate the entire overlapping evaluation dataset every 500 batch steps and at epoch end. The reported mean averages batch means equally, so a short final batch receives too much weight. Empty datasets can also cause division by zero. Evaluation always restores training mode rather than the caller's previous mode.

Fix: accumulate summed negative log likelihood and the number of valid targets, then divide those totals. For frequent monitoring, use a fixed validation subset; perform full evaluation at meaningful intervals. Preserve and restore the prior model mode. Add empty-data validation. Report perplexity only with the tokenizer and scoring protocol specified; token-based perplexities from different tokenizers are not directly comparable. Bits per byte can help comparisons on the same underlying text.

**3.8 Checkpoints cannot resume a training run — P1.**

Where: [checkpoint save](/Users/maskedpirate/repos/transformers/src/train_gpt.py:140).

You save only model weights. Optimizer state, configuration, tokenizer identity, random states, progress, and any future scheduler/scaler are absent. There is no resume path. Because epochs are zero-indexed, the first save is after the sixth completed epoch; a five-epoch run writes no checkpoint. There is no unconditional final save.

Fix: save a versioned checkpoint containing model and optimizer state, model/training configuration, global optimizer step, token count, tokenizer fingerprint, random generator states, and scheduler/scaler/sampling state when used. Save periodically by step or time, retain a best validation checkpoint if useful, and always save the final state. Write atomically so an interrupted write does not replace a valid checkpoint. Verify a resume against an uninterrupted tiny run.

**3.9 The input pipeline will not scale to a large corpus — P2.**

Where: [whole-file reads](/Users/maskedpirate/repos/transformers/src/train_gpt.py:93), [Python start-index list](/Users/maskedpirate/repos/transformers/src/train_gpt.py:40).

The process holds whole strings, Python token lists, an int64 token tensor, and a Python integer list for every start position. This is unnecessary memory overhead. Distributed processes would repeat that preparation independently.

Fix: tokenize once into versioned binary shards with a compact storage dtype appropriate to the vocabulary. Store document offsets and tokenizer/data fingerprints. Memory-map or stream the shards, converting batches to the integer dtype required by embeddings/loss. Compute valid starts arithmetically rather than storing every index. Benchmark shard reading and worker memory before scaling the corpus.

**3.10 There is no performance evidence yet — P1.**

Printing loss and parameter count does not reveal where time or memory goes. Add tokens per second, optimizer-step time, input-wait time, peak GPU memory, and validation loss against processed tokens. On GPU, use correctly synchronized timing or CUDA events. Separate startup, tokenization, training, and evaluation costs. Use a short profiler trace to determine whether attention, feed-forward layers, transfers, logging, or the optimizer dominates. [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

**4. Missing pieces for distributed training**

Multiple DataLoader workers do not train on multiple GPUs. They prepare data for one training process. The repository contains no process-group initialization, DDP/FSDP wrapping, distributed sampler, or collective metric reduction.

For this small model, start with DistributedDataParallel (DDP): each GPU holds a full model, processes different examples, and averages gradients with the other GPUs. Use Fully Sharded Data Parallel (FSDP2) when model, gradient, and optimizer memory become the limiting factor. [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html), [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

| Missing point                          | Why it matters                                                                      | How to fix it                                                                                                                                                                                         |
| -------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Process launch and communication setup | Launching copies today produces independent training jobs.                          | Use `torchrun`, read rank/local rank/world size, initialize a process group, choose a compatible backend, and clean up on exit. Start with one process per CUDA GPU.                                  |
| Explicit local device                  | The current code repeatedly uses bare `cuda` and never selects a local-rank device. | Set the process device, and pass that device consistently. Create masks/buffers from input devices.                                                                                                   |
| Gradient synchronization               | Different replicas would learn different weights.                                   | Wrap the model in DDP and perform forward calls through the wrapper. Move causal-mask handling into the model/attention interface so training does not rely on custom methods exposed by the wrapper. |
| Different data for each rank           | Ordinary `shuffle=True` does not partition data across processes.                   | Use a distributed sampler, disable conflicting loader shuffling, and call `sampler.set_epoch(epoch)`. Partition streaming shards across ranks and workers too.                                        |
| Consistent number of collective calls  | Ranks ending at different times can hang.                                           | Define last-batch and uneven-input policies. Ensure ranks agree on optimizer, evaluation, and checkpoint schedules.                                                                                   |
| Effective-batch accounting             | More GPUs and accumulation change the training regime.                              | Track global batch and token counts. Use DDP `no_sync()` for intermediate accumulation microbatches and synchronize the final one. Scale loss by the intended accumulation/token normalization.       |
| Global validation metrics              | A rank's loss describes only its portion of the data.                               | Sum valid-token losses and counts across ranks. Avoid counting sampler padding/duplicated validation examples twice.                                                                                  |
| Coordinated output and checkpoints     | Every rank could overwrite the same file or flood the console.                      | For ordinary DDP, write logs and replicated checkpoints from rank zero with coordinated control flow. Record rank-specific RNG/sampling state when exact restart matters.                             |
| Restart and failure handling           | A failed worker can leave others waiting or lose hours of progress.                 | Add periodic restartable checkpoints, meaningful timeout/error reporting, and a tested recovery procedure. A launcher alone does not recreate the training state.                                     |

The launch/device details are documented in [torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html). DDP's synchronization and accumulation behavior is documented in [the DDP API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html). Data partitioning and epoch reseeding are covered by [DistributedSampler](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.distributed.DistributedSampler).

For fixed-length unpadded batches, a useful formula is:

`global tokens per optimizer update = local batch size × sequence length × world size × accumulation steps`

At your defaults, two GPUs and four accumulation steps give `64 × 256 × 2 × 4 = 131,072` tokens per update. A partial final batch or padding changes that count. When ranks/microbatches have different valid-token counts, averaging their mean losses equally gives the wrong global weighting; account for DDP's gradient averaging and normalize using the intended total token count.

For a new sharded implementation, use the current FSDP2 `fully_shard` workflow, apply it at appropriate block boundaries, and construct the optimizer after sharding. Sharding saves parameter/gradient/optimizer memory but does not remove activation memory; selective activation checkpointing may still help, at the cost of recomputation. Use Distributed Checkpoint for scalable model/optimizer state and resharding. All ranks must participate in required distributed checkpoint operations; “only rank zero saves” is not a universal FSDP rule. [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Distributed Checkpoint tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html).

Tensor parallelism splits individual large layers, pipeline parallelism splits layers between devices, and context parallelism splits long-sequence work. Expert parallelism distributes MoE experts. These become useful when a specific memory or compute limit requires them; they are not the next step for this model. Study [TorchTitan](https://github.com/pytorch/torchtitan) for how PyTorch's training components fit together, rather than trying to build every parallelism mode simultaneously.

Before claiming distributed support, verify that a tiny two-rank update matches a single-process update on the same global batch within numerical tolerance, that training examples are partitioned as intended, that evaluation counts are correct, and that interruption/resume works. Then measure scaling efficiency at a fixed global batch and clearly describe any separate experiment that increases the batch with GPU count.

**5. PyTorch usage and coding practices in October 2026**

**5.1 Device decisions belong in the application, not model internals.**

Where: [`gpt.py`](/Users/maskedpirate/repos/transformers/src/gpt.py:53), [`train_gpt.py`](/Users/maskedpirate/repos/transformers/src/train_gpt.py:86), [`transformer.py` import side effects](/Users/maskedpirate/repos/transformers/src/transformer.py:8).

Hard-coded CUDA prevents ordinary CPU checks and complicates multiple GPUs. Importing `transformer.py` asks for GPU zero immediately and failed on this CPU environment. It also suppresses warnings globally, concealing useful feedback.

Fix: select a device in the CLI/trainer, pass or infer it from tensors, and make module imports quiet. Remove the global warning suppression. Use `torch.inference_mode()` where suitable for generation and retain clear train/eval behavior. CPU support is valuable for correctness tests even if serious training remains CUDA-based.

**5.2 The dependency policy is inconsistent and includes unused requirements.**

Where: [`pyproject.toml`](/Users/maskedpirate/repos/transformers/pyproject.toml:5).

PyTorch 2.14 was released in September 2026, while your package pins 2.4.1. An older pin is not itself a bug: reproducible research can deliberately use an older version. But it excludes current features and needs an explicit compatibility policy. Other dependencies are unpinned, `tiktoken` is imported unconditionally but not declared, Pydantic v2 methods are used without a v2 requirement, and `flash-attn` is mandatory despite being unused. Python 3.11+ is stated in the README but not in `requires-python`. [Official PyTorch 2.14 announcement](https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431).

Fix: declare the actual runtime requirements, supported Python/PyTorch versions, and Pydantic's needed major version. Make specialized accelerator dependencies optional if native PyTorch attention serves the normal path. Either declare tiktoken or import it only for an explicitly installed tokenizer extra. Use a tested lock/constraints setup for training experiments. Upgrade intentionally with the numerical and CUDA checks required by your chosen environment.

**5.3 Packaging builds, but the module namespace is fragile.**

The wheel build succeeded. Its modules are installed under generic top-level names such as `blocks`, `gpt`, `generate`, and `tokenizer`; the distribution name does not create an import namespace. Those names can collide with other code and make imports ambiguous.

Fix: use a named package under `src/`, package-relative imports, and console entry points. Keep model configuration in a separate module so generation does not need to import the trainer. Test importing the installed wheel from outside the repository.

**5.4 Configuration and CLI behavior need validation.**

Where: [`ModelConfig`](/Users/maskedpirate/repos/transformers/src/train_gpt.py:14), [`main`'s global `args` dependency](/Users/maskedpirate/repos/transformers/src/train_gpt.py:78), [`--shuffle`](/Users/maskedpirate/repos/transformers/src/train_gpt.py:210).

Pydantic currently provides typed defaults but does not enforce the key architectural relationships. An `assert` checks head divisibility later, and assertions can be disabled. `main()` relies on global CLI state. `type=bool` means a string such as `"False"` is truthy, so `--shuffle False` does not behave as users expect.

Fix: validate positive sizes, embedding/head divisibility, even rotary dimensions, allowed positional types, tokenizer vocabulary compatibility, and dropout range. Raise informative errors before allocating a model. Pass configuration objects into `main()`. Use `BooleanOptionalAction` or explicit flags for shuffling. Include dropout, positional choice, learning-rate settings, device, and precision in saved configuration.

**5.5 Tests and experiment records are missing.**

Formatting hooks cannot detect incorrect attention semantics. Add small `unittest` checks for valid Unicode/control-character round trips, empty and one-token decoding, tokenizer save/load identity, unequal-length cross-attention, causal invariance, output/logit shapes, padding behavior, configuration/checkpoint reconstruction, and loss accounting. Add accelerator checks for BF16/FP16 RoPE and attention; skip them clearly when unavailable. A tiny overfit test is useful evidence that the whole training path can learn.

Record the seed, model/configuration, data/tokenizer fingerprints, package versions, hardware, processed tokens, and validation protocol. Seed model initialization and data sampling explicitly. Strict deterministic execution can cost throughput, so provide a reproducibility option and document its scope rather than pretending a seed guarantees identical results on every device.

**5.6 Documentation currently overstates some capabilities.**

The README claims CPU/CUDA support, positional choices, caching, and efficient data loading. The code currently forces CUDA, combines positional methods, does not call the statistics cache, and uses heavily overlapping windows. CLI help calls the custom tokenizer SentencePiece even though it is not SentencePiece.

Fix: document measured behavior and working commands. State device requirements accurately, explain processed-token accounting, and keep architecture/tokenizer names consistent. Separate an educational reference path from an optimized path if you want both to remain readable.

**6. Tokenization correctness and performance**

**6.1 Byte IDs and special-token IDs overlap — P0.**

Where: [`Tokenizer.__init__`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:20).

What is lacking: bytes occupy IDs 0–255, and PAD/UNK/BOS/EOS also occupy IDs 0–3. Adding the specials overwrites the first four byte entries in the vocabulary. Merge IDs start at 260, leaving IDs 256–259 undefined.

Why it matters: my round-trip check turned `\x00A\x01B\x02C\x03D` into `ABCD`; decoding treated real bytes as special tokens and removed them. Merges involving those bytes can also reconstruct the wrong bytes. An output head can predict the undefined IDs and cause a decoding error. Ordinary text without those control bytes may appear to work, which makes the issue easy to miss.

How to fix it: use disjoint spaces, for example bytes 0–255, specials 256–259, and merges starting at 260. Alternatively, shift the entire byte vocabulary consistently. Enforce uniqueness and preserve all 256 bytes. Derive the embedding/output capacity from the valid ID range, not an assumption that dictionary length equals the largest ID plus one. Correcting the scheme changes tokenizer identity: regenerate encoded data and use new compatible checkpoints, or design an explicit migration; do not silently reuse old artifacts.

**6.2 Regex pre-tokenization is immediately undone — P1.**

Where: [`Tokenizer.train`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:140), [`Tokenizer.encode`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:97).

What is lacking: training splits the text with a regex, then joins all pieces back together before counting pairs. Encoding does not apply the regex at all.

Why it matters: boundaries do not constrain merges. In my `"a b"` example, training merged `a` with the following space across the regex boundary. Whole-stream BPE can be an intentional design, but this is not the pre-tokenized design suggested by the code. It also loses the opportunity to process short repeated pieces efficiently.

How to fix it: keep pre-tokenized pieces separate, aggregate pair counts across them, and merge only within each piece. Apply the exact same splitting policy at encoding time. Save the pattern and any normalization policy. For source code or multilingual text, make normalization an explicit choice because changing whitespace or characters can lose information. [Hugging Face Tokenizers quicktour](https://huggingface.co/docs/tokenizers/quicktour).

**6.3 The tokenizer algorithms repeatedly scan Python lists — P1/P2.**

Where: [pair counting](/Users/maskedpirate/repos/transformers/src/tokenizer.py:34), [encoding loop](/Users/maskedpirate/repos/transformers/src/tokenizer.py:99), [training loop](/Users/maskedpirate/repos/transformers/src/tokenizer.py:148).

What is lacking: each training merge recounts and rewrites the token sequence. Encoding repeatedly constructs all pair statistics, chooses a pair, and scans the sequence again. Work grows with both input length and the number of merge rounds. The statistics memoization method is not called; caching entire token sequences would also consume substantial memory with little reuse while training changes the sequence each round.

Why it matters: Python bookkeeping can dominate preprocessing on large text. Faster tokenization will mostly improve preprocessing/startup in this trainer, because tokens are already computed once during dataset construction; it will not independently make the GPU's forward pass faster.

Measured result: I trained 124 merges for a 384-ID target on 19,600 bytes of synthetic repeated text. I gave your encoder and a native tiktoken encoder the same byte/merge ranks and a whole-text pattern matching your current behavior. Both produced identical token ID sequences. After warmup, median times from three runs were:

| Input                       | Your Python encoder | Native encoder | Ratio      |
| --------------------------- | ------------------- | -------------- | ---------- |
| 19,600 bytes; 3,000 tokens  | 136.6 ms            | 0.90 ms        | about 152× |
| 98,000 bytes; 15,000 tokens | 705.4 ms            | 4.90 ms        | about 144× |

These are CPU measurements of a small, highly repetitive synthetic input, using tiktoken 0.14.0. They are not a production corpus benchmark, not a comparison of different vocabularies, and not a guarantee of the same ratio elsewhere. Training this small tokenizer took about 0.144 seconds; I did not compare tokenizer-training throughput with a native trainer.

How to fix it: keep the simple implementation as a learning reference, and use an optimized tokenizer for larger preprocessing. If implementing the optimization yourself, update affected pair counts incrementally, maintain ranked merge candidates, and cache bounded pre-tokenized pieces with genuine reuse. Batch document encoding. Tokenize once into shards. Read the [tiktoken implementation and educational BPE example](https://github.com/openai/tiktoken) and [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/main/en/index).

**6.4 Training and decoding do not handle small inputs robustly — P1.**

Where: [`Tokenizer.train`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:149), [`Tokenizer.decode`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:119), [`--decode` CLI](/Users/maskedpirate/repos/transformers/src/tokenization.py:30).

Training calls `max()` on empty pair counts, so empty or exhausted tiny corpora fail. Vocabulary sizes below the required base are not rejected. Repeated training has no explicit reset/continuation contract. Decoding accepts a tensor even though encoding returns a Python list; list input fails. Squeezing a one-token tensor turns it into a scalar and also fails. The CLI passes a string to the tensor decoder.

Fix: validate the minimum vocabulary and corpus policy, stop merging when no useful pairs remain, and report the actual resulting vocabulary. Define whether retraining resets or extends a tokenizer. Let decoding accept a well-defined sequence of IDs; handle empty/single inputs and define batch behavior separately. Convert a CUDA tensor to a CPU list once rather than inspecting one GPU scalar per token. Parse CLI token IDs as JSON or another explicit format and require a tokenizer path instead of a hard-coded dataset directory.

**6.5 Saved tokenizers omit essential metadata — P1.**

Where: [`save` and `load`](/Users/maskedpirate/repos/transformers/src/tokenizer.py:159).

Only merges and byte vocabulary are saved. The requested/actual size, special-token scheme, pattern, format version, and identity are absent. My tokenizer requested at size 264 loaded with `vocab_size=1024`. It had 260 dictionary entries and maximum ID 263, illustrating the current ID gaps.

Fix: save one versioned format containing the full tokenizer specification, actual ID range, merge order/ranks, and a fingerprint. Preserve bytes explicitly, for example with base64 in JSON. Validate the format on load. Joblib can be acceptable for your own artifacts, but a portable non-executable format is easier to inspect, share, and reproduce. Store the fingerprint in tokenized shards and model checkpoints.

**6.6 The custom and tiktoken training paths are not interchangeable yet — P0/P1.**

Where: [tokenizer selection](/Users/maskedpirate/repos/transformers/src/train_gpt.py:70), [encoding policy](/Users/maskedpirate/repos/transformers/src/train_gpt.py:33), [`generate.py`](/Users/maskedpirate/repos/transformers/src/generate.py:22).

The tiktoken path always selects `cl100k_base`, ignoring the supplied tokenizer name, while the model still defaults to 384 vocabulary entries. That tokenizer can emit IDs far above 383, causing embedding or target errors. Generation only loads your custom tokenizer. `allowed_special="all"` also interprets recognized special-token strings in ordinary corpus text, while custom encoding treats those strings as ordinary bytes.

Fix: provide one tokenizer interface with explicit vocabulary capacity, valid/special IDs, encoding, decoding, and identity. Select the requested encoding, derive and verify model dimensions, and reconstruct the same backend during generation. Choose whether special strings are ordinary text or control markers; insert document control IDs deliberately. When a tokenizer has gaps in its ID range, output-head sizing and legal generation IDs need separate handling. [tiktoken API and source](https://github.com/openai/tiktoken).

**6.7 Vocabulary quality is unmeasured — P2.**

BPE itself is not outdated; modern model reports still describe byte-level BPE. The question is whether its vocabulary suits your data. A 384-ID vocabulary leaves only 124 learned merges after a correct 256-byte/four-special base. That can be reasonable for a toy model, but can make real text consume many more tokens, especially outside the training language.

Fix: evaluate bytes or characters per token, token-length distributions, round-trip fidelity, and compression separately for English, Turkish, code, and other target content. Measure encoding throughput and training time at several vocabulary sizes. A larger vocabulary shortens sequences but enlarges embeddings and the output/loss computation; choose using end-to-end memory, speed, and validation quality. Train the tokenizer on representative training data, then freeze it for evaluation. [Qwen3's documented byte-level BPE](https://arxiv.org/html/2505.09388v1#S2).

**7. Suggested improvement sequence and evidence to collect**

1. **Establish correctness.** Repair token IDs, position handling, encoder–decoder attention/masks/logits, device assumptions, and checkpoint/model reconstruction. Run the small correctness checks and a tiny overfit experiment. Keep an explicitly documented reference implementation.
1. **Make training interpretable.** Replace the default overlapping epoch scheme with packed blocks or a defined sampling budget. Add configuration/seed/data records, token-weighted validation, and complete restartable checkpoints. Demonstrate resume equivalence on a tiny run.
1. **Measure single-GPU performance.** Compare the corrected eager FP32 baseline with SDPA, then BF16/FP16, then compilation and optimizer changes. Record tokens/second, peak memory, and loss versus processed tokens. Include preprocessing separately. Check numerical behavior after each change.
1. **Modernize the dense block.** Compare pre-normalization/RMSNorm and SwiGLU at a similar parameter/compute budget. Add GQA after implementing KV caching if generation memory is important. Change one major choice at a time so results have a clear cause.
1. **Add distributed correctness.** Implement two-rank DDP, partitioned data, accumulation, global evaluation, coordinated checkpoints, and resume. Verify equivalence before measuring scaling. Introduce FSDP2 only when replicated state memory becomes a real limit.
1. **Choose a research target.** Longer context, MoE, hybrid attention, FP8, and multi-token prediction should answer a specific measured limitation. Report both quality and resource cost.

**8. Reading route**

| Read                                                                                                                                                                                                                                                                       | What to look for                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)                                                                                                                                                                                                           | Query/key/value roles, encoder–decoder attention, causal masking, residual blocks, and sinusoidal formulas. |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)                                                                                                                                                                                 | Why moving normalization changes optimization and warmup sensitivity.                                       |
| [RoFormer](https://arxiv.org/abs/2104.09864)                                                                                                                                                                                                                               | How rotary positions act on attention queries and keys.                                                     |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)                                                                                                                                                                                                       | Gated feed-forward networks and fair parameter-budget comparisons.                                          |
| [GQA](https://arxiv.org/abs/2305.13245)                                                                                                                                                                                                                                    | The quality/memory tradeoff when query heads share keys and values.                                         |
| [Qwen3 architecture section](https://arxiv.org/html/2505.09388v1#S2)                                                                                                                                                                                                       | A concrete modern dense baseline with clearly specified design choices.                                     |
| [Qwen3.8-27B official model card](https://huggingface.co/Qwen/Qwen3.8-27B)                                                                                                                                                                                                 | A 2026 example of hybrid attention and the distinction between architecture and post-training.              |
| [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)                                                                                                                                                                                                           | Advanced compressed attention, MoE, residual design, optimizer choices, and long-context tradeoffs.         |
| [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)                                                                                                                                                         | Kernel dispatch, causal alignment, boolean-mask meaning, and dropout behavior.                              |
| [PyTorch AMP](https://docs.pytorch.org/docs/2.14/amp.html)                                                                                                                                                                                                                 | Deliberate precision policies and gradient scaling.                                                         |
| [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)                                                                                   | Measuring real bottlenecks before selecting optimizations.                                                  |
| [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html), [torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html), and [DistributedSampler](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.distributed.DistributedSampler) | The first complete multi-GPU training path.                                                                 |
| [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)                                                                                    | State sharding and restartable larger-model training.                                                       |
| [TorchTitan](https://github.com/pytorch/torchtitan)                                                                                                                                                                                                                        | A maintained reference for composing PyTorch training and parallelism features.                             |
| [Tokenizers quicktour](https://huggingface.co/docs/tokenizers/quicktour) and [tiktoken](https://github.com/openai/tiktoken)                                                                                                                                                | Pre-tokenization boundaries, special tokens, serialization, batching, and native BPE implementations.       |

The most useful immediate outcome would be a small model whose tokenizer is lossless, attention is correct, checkpoints reconstruct it automatically, and reported training tokens mean what they say. That would give you a sound platform for learning from modern architecture and systems experiments.
