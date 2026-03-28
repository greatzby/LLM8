#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PAD_TOKEN = "<pad>"
NEWLINE_TOKEN = "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare prompt-aware textualized GraphA/GraphNano dataset into fixed-length .bin files."
    )
    p.add_argument("--data_dir", type=Path, required=True, help="Dataset dir containing train_K.txt / test.txt")
    p.add_argument("--train_paths_per_pair", type=int, default=20, help="Use train_{K}.txt")
    p.add_argument(
        "--block_multiple",
        type=int,
        default=64,
        help="Round block_size up to a multiple of this value",
    )
    p.add_argument(
        "--pad_token",
        type=str,
        default=PAD_TOKEN,
        help="Padding token to use in the fixed-length serialization",
    )
    return p.parse_args()


def read_nonempty_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def file_char_length(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return len(f.read())


def split_tokens(line: str) -> List[str]:
    return line.strip().split()


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"block_multiple must be positive, got {multiple}")
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def load_instruction_template(data_dir: Path) -> Tuple[Optional[List[str]], Optional[str]]:
    tmpl_file = data_dir / "instruction_template.json"
    if not tmpl_file.exists():
        return None, None

    with open(tmpl_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    toks = obj.get("prompt_template_tokens")
    text = obj.get("prompt_template_text")

    if isinstance(toks, list) and len(toks) > 0:
        toks = [str(x) for x in toks]
        if not text:
            text = " ".join(toks)
        return toks, text

    return None, text


def collect_vocab(
    train_lines: Sequence[str],
    test_lines: Sequence[str],
    pad_token: str,
) -> Tuple[Dict[str, int], Dict[int, str], List[str], List[str]]:
    observed = set()
    for line in list(train_lines) + list(test_lines):
        observed.update(split_tokens(line))

    numeric_tokens = sorted([tok for tok in observed if tok.isdigit()], key=lambda x: int(x))
    word_tokens = sorted([
        tok for tok in observed
        if (not tok.isdigit()) and tok not in {pad_token, NEWLINE_TOKEN}
    ])

    vocab = [pad_token, NEWLINE_TOKEN] + word_tokens + numeric_tokens

    stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab)}
    itos: Dict[int, str] = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos, numeric_tokens, word_tokens


def encode_line_fixed(
    line: str,
    stoi: Dict[str, int],
    block_size: int,
    pad_token: str,
) -> List[int]:
    toks = split_tokens(line)

    if len(toks) > block_size:
        raise ValueError(
            f"Sequence length {len(toks)} exceeds block_size={block_size}. "
            "Increase --block_multiple or shorten prompts."
        )

    ids = [stoi[tok] for tok in toks]
    ids.append(stoi[NEWLINE_TOKEN])

    total_len = block_size + 1
    if len(ids) > total_len:
        raise ValueError(
            f"Encoded sequence length {len(ids)} exceeds fixed span {total_len}."
        )

    pad_id = stoi[pad_token]
    ids.extend([pad_id] * (total_len - len(ids)))
    return ids


def write_bin(
    lines: Sequence[str],
    out_path: Path,
    stoi: Dict[str, int],
    block_size: int,
    pad_token: str,
) -> None:
    seq_span = block_size + 1
    total_tokens = len(lines) * seq_span
    arr = np.empty(total_tokens, dtype=np.uint16)

    pos = 0
    for line in lines:
        ids = encode_line_fixed(line, stoi, block_size, pad_token)
        arr[pos: pos + seq_span] = np.asarray(ids, dtype=np.uint16)
        pos += seq_span

    arr.tofile(out_path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"

    if not train_txt.exists():
        raise FileNotFoundError(f"Missing training text file: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Missing test text file: {test_txt}")

    train_lines = read_nonempty_lines(train_txt)
    test_lines = read_nonempty_lines(test_txt)

    if not train_lines:
        raise ValueError(f"No non-empty lines found in {train_txt}")
    if not test_lines:
        raise ValueError(f"No non-empty lines found in {test_txt}")

    train_text_len = file_char_length(train_txt)
    test_text_len = file_char_length(test_txt)

    max_train_toks = max(len(split_tokens(line)) for line in train_lines)
    max_test_toks = max(len(split_tokens(line)) for line in test_lines)
    max_seq_toks = max(max_train_toks, max_test_toks)

    block_size = round_up_to_multiple(max_seq_toks, args.block_multiple)

    stoi, itos, numeric_tokens, word_tokens = collect_vocab(
        train_lines=train_lines,
        test_lines=test_lines,
        pad_token=args.pad_token,
    )

    vocab_size = len(stoi)
    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Vocabulary too large for uint16 storage: vocab_size={vocab_size}"
        )

    prompt_template_tokens, prompt_template_text = load_instruction_template(data_dir)

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"

    write_bin(train_lines, train_bin, stoi, block_size, args.pad_token)
    write_bin(test_lines, val_bin, stoi, block_size, args.pad_token)

    meta = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "stoi": stoi,
        "itos": itos,
        "pad_token": args.pad_token,
        "newline_token": NEWLINE_TOKEN,
        "train_text_file": train_txt.name,
        "val_text_file": test_txt.name,
        "train_bin_file": train_bin.name,
        "val_bin_file": val_bin.name,
        "train_paths_per_pair": args.train_paths_per_pair,
        "prompt_template_tokens": prompt_template_tokens,
        "prompt_template_text": prompt_template_text,
        "num_numeric_tokens": len(numeric_tokens),
        "num_word_tokens": len(word_tokens),
        "max_train_tokens": max_train_toks,
        "max_test_tokens": max_test_toks,
    }

    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("=" * 70)
    print(f"Prepared dataset in: {data_dir}")
    print(f"  train text len : {train_text_len:,}")
    print(f"  test  text len : {test_text_len:,}")
    print(f"  vocab size     : {vocab_size}")
    print(f"  #numeric toks  : {len(numeric_tokens)}")
    print(f"  #word toks     : {len(word_tokens)}")
    print(f"  max train toks : {max_train_toks}")
    print(f"  max test  toks : {max_test_toks}")
    print(f"  block size     : {block_size}")
    print(f"  wrote          : {train_bin.name}, {val_bin.name}, meta.pkl")
    if prompt_template_text:
        print(f"  prompt template: {prompt_template_text}")
    else:
        print("  prompt template: [not found]")
    print("=" * 70)


if __name__ == "__main__":
    main()