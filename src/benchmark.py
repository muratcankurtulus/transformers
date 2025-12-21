#!/usr/bin/env python3
"""Benchmark script for tokenizer and pretokenization performance.

Measures:
- Tokenization throughput (tokens/second)
- Peak RAM usage
- Encoding time for various text sizes
"""

import argparse
import gc
import os
import resource
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import Tokenizer


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB (Linux/macOS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux, bytes on macOS
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1024 / 1024
    else:
        return usage.ru_maxrss / 1024


def benchmark_encode(tokenizer: Tokenizer, text: str, iterations: int = 3) -> dict:
    """Benchmark encoding performance.

    Args:
        tokenizer: Trained tokenizer
        text: Text to encode
        iterations: Number of iterations for averaging

    Returns:
        Dict with benchmark results
    """
    text_bytes = len(text.encode("utf-8"))
    results = []

    # Clear cache before benchmarking
    tokenizer.clear_cache()
    gc.collect()

    # Warmup run
    _ = tokenizer.encode(text[: min(10000, len(text))])

    for i in range(iterations):
        # Clear cache between runs for fair comparison
        tokenizer.clear_cache()
        gc.collect()

        start_time = time.perf_counter()
        tokens = tokenizer.encode(text)
        elapsed = time.perf_counter() - start_time

        results.append(
            {
                "elapsed_seconds": elapsed,
                "num_tokens": len(tokens),
                "tokens_per_second": len(tokens) / elapsed,
                "bytes_per_second": text_bytes / elapsed,
            }
        )

        print(f"  Run {i+1}: {len(tokens):,} tokens in {elapsed:.2f}s ({len(tokens)/elapsed:,.0f} tok/s)")

    # Average results
    avg_results = {
        "text_bytes": text_bytes,
        "text_chars": len(text),
        "iterations": iterations,
        "avg_elapsed_seconds": sum(r["elapsed_seconds"] for r in results) / iterations,
        "avg_num_tokens": sum(r["num_tokens"] for r in results) / iterations,
        "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results) / iterations,
        "avg_bytes_per_second": sum(r["bytes_per_second"] for r in results) / iterations,
        "peak_memory_mb": get_peak_memory_mb(),
    }

    return avg_results


def benchmark_encode_cached(tokenizer: Tokenizer, text: str, iterations: int = 3) -> dict:
    """Benchmark encoding with cache (realistic repeated-text scenario).

    Args:
        tokenizer: Trained tokenizer
        text: Text to encode
        iterations: Number of iterations

    Returns:
        Dict with benchmark results
    """
    len(text.encode("utf-8"))
    results = []

    # Clear cache and do warmup
    tokenizer.clear_cache()
    gc.collect()

    for i in range(iterations):
        # Don't clear cache - measure cached performance
        start_time = time.perf_counter()
        tokens = tokenizer.encode(text)
        elapsed = time.perf_counter() - start_time

        results.append(
            {
                "elapsed_seconds": elapsed,
                "num_tokens": len(tokens),
                "tokens_per_second": len(tokens) / elapsed,
            }
        )

        print(f"  Run {i+1} (cached): {len(tokens):,} tokens in {elapsed:.2f}s ({len(tokens)/elapsed:,.0f} tok/s)")

    avg_results = {
        "avg_elapsed_seconds": sum(r["elapsed_seconds"] for r in results) / iterations,
        "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results) / iterations,
    }

    return avg_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenizer performance")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file for benchmarking",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for averaging (default: 3)",
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=None,
        help="Maximum bytes to read from input file",
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    try:
        tokenizer = Tokenizer.load(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Merges: {len(tokenizer.merges)}")

    # Load text
    print(f"\nLoading text from {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        if args.max_bytes:
            text = f.read(args.max_bytes)
        else:
            text = f.read()

    text_bytes = len(text.encode("utf-8"))
    print(f"  Text size: {text_bytes / 1024 / 1024:.2f} MB ({len(text):,} chars)")

    # Run benchmarks
    print(f"\n{'='*60}")
    print("BENCHMARK: Uncached encoding (cold start)")
    print(f"{'='*60}")
    uncached_results = benchmark_encode(tokenizer, text, args.iterations)

    print(f"\n{'='*60}")
    print("BENCHMARK: Cached encoding (warm cache)")
    print(f"{'='*60}")
    cached_results = benchmark_encode_cached(tokenizer, text, args.iterations)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"  Size: {uncached_results['text_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Chars: {uncached_results['text_chars']:,}")
    print(f"  Tokens: {uncached_results['avg_num_tokens']:,.0f}")
    print(f"  Compression ratio: {uncached_results['text_bytes'] / uncached_results['avg_num_tokens']:.2f} bytes/token")
    print()
    print("Uncached performance:")
    print(f"  Time: {uncached_results['avg_elapsed_seconds']:.2f}s")
    print(f"  Throughput: {uncached_results['avg_tokens_per_second']:,.0f} tokens/second")
    print(f"  Throughput: {uncached_results['avg_bytes_per_second'] / 1024 / 1024:.2f} MB/second")
    print()
    print("Cached performance:")
    print(f"  Time: {cached_results['avg_elapsed_seconds']:.2f}s")
    print(f"  Throughput: {cached_results['avg_tokens_per_second']:,.0f} tokens/second")
    print(f"  Speedup: {cached_results['avg_tokens_per_second'] / uncached_results['avg_tokens_per_second']:.1f}x")
    print()
    print(f"Peak RAM: {uncached_results['peak_memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()
