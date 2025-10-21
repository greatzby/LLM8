#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare GraphA Tier-3 dataset for GPT training.

Usage example:
    python data/simple_graph/prepare_compositionnew.py \
        --data_dir data/datasets/graphA_pg020_tier3 \
        --total_nodes 90 \
        --train_paths_per_pair 20 \
        --block_multiple 32
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare composition dataset binaries.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train_X.txt/test.txt.")
    parser.add_argument("--total_nodes", type=int, default=90,
                        help="Total number of node tokens (s1+s2+s3).")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="Used to find train_{K}.txt.")
    parser.add_argument("--block_multiple", type=int, default=32,
                        help="Round block size up to a multiple of this value.")
    return parser.parse_args()


def build_vocab(total_nodes: int) -> tuple[Dict[str, int], Dict[int, str]]:
    stoi = {str(i): i + 2 for i in range(total_nodes)}  # reserve 0,1
    stoi["[PAD]"] = 0
    stoi["\n"] = 1

    itos = {idx: token for token, idx in stoi.items()}
    return stoi, itos


def get_max_token_length(text: str) -> int:
    max_len = 0
    for line in text.strip().split("\n"):
        if line:
            max_len = max(max_len, len(line.strip().split()))
    return max_len


def encode_line(line: str, stoi: Dict[str, int]) -> List[int]:
    tokens = line.strip().split()
    encoded = []
    for token in tokens:
        if token not in stoi:
            raise KeyError(f"Token '{token}' missing in vocabulary.")
        encoded.append(stoi[token])
    encoded.append(stoi["\n"])
    return encoded


def pad_sequence(seq: List[int], block_size: int, pad_id: int) -> List[int]:
    padding = [pad_id] * (block_size - len(seq))
    return seq + padding


def process_file(text: str, stoi: Dict[str, int], block_size: int) -> np.ndarray:
    sequences: List[int] = []
    pad_id = stoi["[PAD]"]

    for line in text.strip().split("\n"):
        if not line:
            continue
        encoded = encode_line(line, stoi)
        if len(encoded) > block_size:
            raise ValueError(
                f"Encoded sequence length {len(encoded)} exceeds block size {block_size}. "
                "Increase --block_multiple or check data."
            )
        padded = pad_sequence(encoded, block_size, pad_id)
        sequences.extend(padded)

    return np.array(sequences, dtype=np.uint16)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"

    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test file not found: {test_txt}")

    train_text = train_txt.read_text(encoding="utf-8")
    test_text = test_txt.read_text(encoding="utf-8")

    print("=" * 70)
    print(f"Preparing dataset in: {data_dir}")
    print(f"Train text length: {len(train_text):,} characters")
    print(f"Test  text length: {len(test_text):,} characters")

    stoi, itos = build_vocab(args.total_nodes)
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size} (includes PAD + newline)")

    max_train = get_max_token_length(train_text)
    max_test = get_max_token_length(test_text)
    max_tokens = max(max_train, max_test) + 1  # include newline token

    block_size = ((max_tokens // args.block_multiple) + 1) * args.block_multiple
    print(f"Max token length (without padding): train={max_train}, test={max_test}")
    print(f"Using block size: {block_size} (multiple of {args.block_multiple})")

    train_ids = process_file(train_text, stoi, block_size)
    val_ids = process_file(test_text, stoi, block_size)

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"

    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    meta = {
        "simple_format": True,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "train_paths_per_pair": args.train_paths_per_pair,
        "total_nodes": args.total_nodes,
    }

    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("=" * 70)
    print("Finished preparing dataset:")
    print(f"  - {train_bin.name}")
    print(f"  - {val_bin.name}")
    print(f"  - meta.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()