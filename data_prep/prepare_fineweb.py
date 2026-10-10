"""Stream a bounded, reproducible English corpus without downloading the full dataset."""

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path


def normalize_document(text: str) -> str:
    """Keep case and paragraph boundaries; remove empty/whitespace-only lines."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def write_corpus(records, output_dir: Path, train_bytes: int, eval_bytes: int, tokenizer_bytes: int) -> dict:
    """Hash whole documents into disjoint splits and bound all output sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in ("train.txt", "eval.txt", "tokenizer_sample.txt", "manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output already exists; choose a new --output_dir to preserve the previous corpus.")
    counts = {"train_bytes": 0, "eval_bytes": 0, "tokenizer_bytes": 0, "train_documents": 0, "eval_documents": 0}
    seen = set()
    with paths[0].open("wb") as train, paths[1].open("wb") as evaluation, paths[2].open("wb") as sample:
        for record in records:
            text = normalize_document(record["text"])
            if not 200 <= len(text) <= 50000:
                continue
            encoded = text.encode("utf-8")
            digest = hashlib.sha256(encoded).digest()
            if digest in seen:
                continue
            seen.add(digest)
            # Split before sampling/tokenizer training; exact duplicates cannot cross splits.
            split = "eval" if int.from_bytes(digest[:8], "big") % 100 == 0 else "train"
            limit = eval_bytes if split == "eval" else train_bytes
            if counts[f"{split}_bytes"] >= limit:
                continue
            encoded += b"\n\n"
            (evaluation if split == "eval" else train).write(encoded)
            counts[f"{split}_bytes"] += len(encoded)
            counts[f"{split}_documents"] += 1
            if split == "train" and counts["tokenizer_bytes"] < tokenizer_bytes:
                sample.write(encoded)
                counts["tokenizer_bytes"] += len(encoded)
            total_documents = counts["train_documents"] + counts["eval_documents"]
            if total_documents % 1000 == 0:
                print(f"{total_documents:,} documents; train={counts['train_bytes'] / 1e6:.1f} MB", flush=True)
            if counts["train_bytes"] >= train_bytes and counts["eval_bytes"] >= eval_bytes:
                break
    if not counts["train_documents"] or not counts["eval_documents"]:
        raise ValueError(
            "Source ended without a nonempty train/evaluation split. Choose larger limits or another source."
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("toy_data/fineweb"))
    parser.add_argument("--train_bytes", type=int, default=300000000)
    parser.add_argument("--eval_bytes", type=int, default=2000000)
    parser.add_argument("--tokenizer_bytes", type=int, default=3000000)
    parser.add_argument("--revision", default="main", help="Dataset commit or branch; resolved commit is recorded")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    args = parser.parse_args()
    if min(args.train_bytes, args.eval_bytes, args.tokenizer_bytes, args.shuffle_buffer) <= 0:
        parser.error("Byte limits and shuffle buffer must be positive")
    # The source is public. Do not consult locally stored Hugging Face credentials.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    cache_dir = args.output_dir.resolve() / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_TOKEN_PATH"] = os.devnull
    os.environ["NETRC"] = os.devnull
    import requests
    from datasets import load_dataset
    from huggingface_hub import HfApi, configure_http_backend

    def public_session():
        session = requests.Session()
        session.trust_env = False  # Prevent Requests from consulting a local .netrc file.
        return session

    configure_http_backend(backend_factory=public_session)

    dataset_id = "HuggingFaceFW/fineweb-edu"
    revision = HfApi().dataset_info(dataset_id, revision=args.revision, token=False).sha
    records = load_dataset(dataset_id, "sample-10BT", split="train", streaming=True, revision=revision, token=False)
    records = records.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    counts = write_corpus(records, args.output_dir, args.train_bytes, args.eval_bytes, args.tokenizer_bytes)
    manifest = {
        "dataset": dataset_id,
        "config": "sample-10BT",
        "revision": revision,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "split": "SHA256(normalized document) modulo 100; bucket 0 is validation",
        "normalization": "NFC, preserve case and paragraph lines, discard documents outside 200..50000 characters",
        "separator": "two newlines; no EOS token inserted",
        "limits_may_exceed_by": "one complete document",
        **counts,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
