#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer


def extract_numbers_first_line(text: str):
    first = text.split("\n", 1)[0]
    return [int(x) for x in re.findall(r"\d+", first)]


def trim_tokens(arr: np.ndarray, pad_id: int, eos_id: int | None):
    # arr is 1D np array of token ids (length = seq_len)
    toks = arr.astype(np.int64).tolist()

    # cut at first PAD (common for fixed-length storage)
    if pad_id in toks:
        toks = toks[: toks.index(pad_id)]

    # optionally cut at EOS (if present before PAD)
    if eos_id is not None and eos_id in toks:
        toks = toks[: toks.index(eos_id) + 1]

    return toks


def classify_pair(source: int, target: int, S1: set[int], S2: set[int], S3: set[int]) -> str:
    if source in S1 and target in S2:
        return "S1->S2"
    if source in S2 and target in S3:
        return "S2->S3"
    if source in S1 and target in S3:
        return "S1->S3"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--train_paths_per_pair", type=int, default=20, help="used to pick train_{K}.bin")
    ap.add_argument("--max_sequences", type=int, default=0, help="0 = scan all sequences, else scan first N")
    ap.add_argument("--show_examples", type=int, default=3, help="print up to N decoded examples per bucket")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()

    # load stages
    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages = stage_info["stages"]
    S1, S2, S3 = stages
    S1s, S2s, S3s = set(map(int, S1)), set(map(int, S2)), set(map(int, S3))

    # load meta
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    seq_len = int(meta["seq_len"])
    pad_id = int(meta["pad_token_id"])
    eos_id = meta.get("eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else None

    # load tokenizer (MUST be local saved)
    tok_dir = data_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=bool(meta.get("trust_remote_code", False)))

    # pick bin
    if args.split == "train":
        bin_path = data_dir / f"train_{args.train_paths_per_pair}.bin"
    else:
        bin_path = data_dir / "val.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing bin file: {bin_path}")

    data = np.memmap(bin_path, dtype=np.uint32, mode="r")
    num_sequences = len(data) // seq_len
    if num_sequences <= 0:
        raise RuntimeError(f"No sequences found. len(data)={len(data)}, seq_len={seq_len}")

    scan_n = num_sequences if args.max_sequences == 0 else min(num_sequences, args.max_sequences)

    counts = Counter()
    bad_parse = 0
    examples = defaultdict(list)

    for i in range(scan_n):
        arr = data[i * seq_len : (i + 1) * seq_len]
        toks = trim_tokens(arr, pad_id=pad_id, eos_id=eos_id)

        if not toks:
            bad_parse += 1
            continue

        text = tokenizer.decode(toks, skip_special_tokens=True)
        nums = extract_numbers_first_line(text)

        if len(nums) < 2:
            bad_parse += 1
            continue

        source, target = int(nums[0]), int(nums[1])
        bucket = classify_pair(source, target, S1s, S2s, S3s)
        counts[bucket] += 1

        if len(examples[bucket]) < args.show_examples:
            examples[bucket].append(text.split("\n", 1)[0])

    total_ok = sum(counts.values())
    print(f"BIN: {bin_path}")
    print(f"seq_len={seq_len}, pad_id={pad_id}, eos_id={eos_id}")
    print(f"num_sequences={num_sequences}, scanned={scan_n}")
    print(f"parsed_ok={total_ok}, bad_parse={bad_parse}\n")

    for k in ["S1->S2", "S2->S3", "S1->S3", "OTHER"]:
        c = counts.get(k, 0)
        pct = (100.0 * c / total_ok) if total_ok else 0.0
        print(f"{k:6s}: {c:8d}  ({pct:6.2f}%)")
        for ex in examples.get(k, []):
            print(f"  ex: {ex}")
        if examples.get(k):
            print()

    # direct answer to your suspicion
    print("----")
    if counts.get("S1->S3", 0) == 0:
        print("结论：扫描范围内未发现任何 S1->S3 训练样本（按 source/target 分桶）。")
    else:
        print(f"结论：发现 S1->S3 样本数 = {counts['S1->S3']}（按 source/target 分桶）。")


if __name__ == "__main__":
    main()