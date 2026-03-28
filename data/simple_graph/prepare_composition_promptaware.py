#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare GraphA dataset binaries (symbolic or instruction-style).")
    p.add_argument("--data_dir", type=Path, required=True, help="Directory containing train_K.txt / test.txt")
    p.add_argument("--train_paths_per_pair", type=int, default=20)
    p.add_argument("--block_multiple", type=int, default=64,
                   help="Round block size up to a multiple of this value. I recommend 64 for this pilot.")
    return p.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def collect_tokens(texts: List[str], extra_non_digit_tokens: List[str] | None = None) -> Tuple[List[str], List[str]]:
    numeric_tokens: Set[str] = set()
    word_tokens: Set[str] = set()

    for text in texts:
        for line in text.strip().splitlines():
            for tok in line.strip().split():
                if tok.isdigit():
                    numeric_tokens.add(tok)
                else:
                    word_tokens.add(tok)

    if extra_non_digit_tokens:
        for tok in extra_non_digit_tokens:
            if tok not in {"{s}", "{t}"} and not tok.isdigit():
                word_tokens.add(tok)

    numeric_sorted = sorted(numeric_tokens, key=lambda x: int(x))
    word_sorted = sorted(word_tokens)
    return numeric_sorted, word_sorted


def build_vocab(numeric_tokens: List[str], word_tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    stoi: Dict[str, int] = {
        "[PAD]": 0,
        "\n": 1,
    }
    idx = 2

    for tok in numeric_tokens:
        stoi[tok] = idx
        idx += 1

    for tok in word_tokens:
        if tok in stoi:
            continue
        stoi[tok] = idx
        idx += 1

    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


def get_max_token_length(text: str) -> int:
    max_len = 0
    for line in text.strip().splitlines():
        toks = line.strip().split()
        if toks:
            max_len = max(max_len, len(toks))
    return max_len


def encode_line(line: str, stoi: Dict[str, int]) -> List[int]:
    toks = line.strip().split()
    out = []
    for tok in toks:
        if tok not in stoi:
            raise KeyError(f"Token '{tok}' missing in vocabulary.")
        out.append(stoi[tok])
    out.append(stoi["\n"])
    return out


def pad_sequence(seq: List[int], block_size: int, pad_id: int) -> List[int]:
    if len(seq) > block_size:
        raise ValueError(f"Encoded sequence length {len(seq)} exceeds block_size={block_size}")
    return seq + [pad_id] * (block_size - len(seq))


def process_file(text: str, stoi: Dict[str, int], block_size: int) -> np.ndarray:
    flat: List[int] = []
    pad_id = stoi["[PAD]"]
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        encoded = encode_line(line, stoi)
        flat.extend(pad_sequence(encoded, block_size, pad_id))
    return np.array(flat, dtype=np.uint16)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"

    train_text = read_text(train_txt)
    test_text = read_text(test_txt)

    template_file = data_dir / "instruction_template.json"
    prompt_template_tokens = None
    extra_non_digit_tokens = []
    if template_file.exists():
        with open(template_file, "r", encoding="utf-8") as f:
            tmpl = json.load(f)
        prompt_template_tokens = tmpl.get("prompt_template_tokens")
        extra_non_digit_tokens = prompt_template_tokens or []

    numeric_tokens, word_tokens = collect_tokens(
        [train_text, test_text],
        extra_non_digit_tokens=extra_non_digit_tokens,
    )

    stoi, itos = build_vocab(numeric_tokens, word_tokens)
    vocab_size = len(stoi)

    max_train = get_max_token_length(train_text)
    max_test = get_max_token_length(test_text)
    max_tokens = max(max_train, max_test) + 1  # + newline token

    # keep behavior similar to your old script
    block_size = ((max_tokens // args.block_multiple) + 1) * args.block_multiple

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
        "prompt_template_tokens": prompt_template_tokens,
        "numeric_tokens": numeric_tokens,
        "word_tokens": word_tokens,
    }

    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("=" * 70)
    print(f"Prepared dataset in: {data_dir}")
    print(f"  train text len : {len(train_text):,}")
    print(f"  test  text len : {len(test_text):,}")
    print(f"  vocab size     : {vocab_size}")
    print(f"  #numeric toks  : {len(numeric_tokens)}")
    print(f"  #word toks     : {len(word_tokens)}")
    print(f"  max train toks : {max_train}")
    print(f"  max test  toks : {max_test}")
    print(f"  block size     : {block_size}")
    print(f"  wrote          : {train_bin.name}, {val_bin.name}, meta.pkl")
    if prompt_template_tokens is not None:
        print(f"  prompt template: {' '.join(prompt_template_tokens)}")
    print("=" * 70)


if __name__ == "__main__":
    main()