#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment GraphA Tier-3 datasets with partial-path (prefix/suffix) samples.

This script is intended to be run AFTER generating a P13-only dataset.
It creates additional training lines of the form:
    s t s v1 v2 ... vk   (prefix crop, no target at the end)
    s t vk ... t         (suffix crop, starts mid-way, ends at target)
The original lines are kept (unless --keep-original 0 is specified).

Example usage:

  python augment_partial_paths.py \
      --src-dir data/datasets/graphA_pg030_tier3_P13_100 \
      --dest-dir data/datasets/graphA_pg030_tier3_P13_100_crops \
      --paths-per-pair 20 \
      --prefix-samples 1 \
      --suffix-samples 1 \
      --min-prefix-len 3 \
      --max-prefix-len 6 \
      --min-suffix-len 3 \
      --max-suffix-len 6 \
      --seed 2025

After running, use prepare_compositionnew.py and your training script
on the new destination directory as usual.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_COPY_ITEMS = [
    "composition_graph.graphml",
    "dataset_summary.json",
    "stage_info.pkl",
    "test.txt",
    "val.bin",
    "meta.pkl",
    "train_{paths}.bin",
    "p13_control_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment GraphA Tier-3 dataset with cropped path samples."
    )
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="Source dataset directory (after P13 control).")
    parser.add_argument("--dest-dir", type=Path, required=True,
                        help="Destination directory for augmented dataset.")
    parser.add_argument("--paths-per-pair", type=int, default=20,
                        help="Used to locate train_{K}.txt.")
    parser.add_argument("--train-file-pattern", type=str, default="train_{paths}.txt",
                        help="Format string for the training file name.")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility.")

    parser.add_argument("--keep-original", type=int, default=1,
                        help="If 1, keep original lines; if 0, only cropped samples.")

    parser.add_argument("--prefix-samples", type=int, default=1,
                        help="Number of prefix crops per original line (0 to disable).")
    parser.add_argument("--suffix-samples", type=int, default=1,
                        help="Number of suffix crops per original line (0 to disable).")

    parser.add_argument("--min-prefix-len", type=int, default=3,
                        help="Minimum number of nodes in prefix crop (excluding the leading src/dst pair).")
    parser.add_argument("--max-prefix-len", type=int, default=6,
                        help="Maximum number of nodes in prefix crop.")
    parser.add_argument("--min-suffix-len", type=int, default=3,
                        help="Minimum number of nodes (including target) in suffix crop.")
    parser.add_argument("--max-suffix-len", type=int, default=6,
                        help="Maximum number of nodes in suffix crop.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle final lines before writing.")
    parser.add_argument("--copy-patterns", nargs="*",
                        default=DEFAULT_COPY_ITEMS,
                        help="File patterns to copy from src to dest if present.")
    return parser.parse_args()


def load_train_lines(train_path: Path) -> List[str]:
    with train_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def choose_lengths(
    candidates: List[int],
    count: int,
    rng: random.Random,
) -> List[int]:
    if not candidates or count <= 0:
        return []
    if count >= len(candidates):
        return candidates
    return rng.sample(candidates, count)


def generate_prefix_crops(
    path_nodes: Sequence[str],
    prefix_samples: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> List[List[str]]:
    upper = min(max_len, max(0, len(path_nodes) - 1))
    candidates = [l for l in range(min_len, upper + 1) if l < len(path_nodes)]
    lengths = choose_lengths(candidates, prefix_samples, rng)
    return [list(path_nodes[:length]) for length in lengths]


def generate_suffix_crops(
    path_nodes: Sequence[str],
    suffix_samples: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> List[List[str]]:
    upper = min(max_len, max(0, len(path_nodes) - 1))
    candidates = [l for l in range(min_len, upper + 1) if l < len(path_nodes)]
    lengths = choose_lengths(candidates, suffix_samples, rng)
    return [list(path_nodes[-length:]) for length in lengths]


def copy_side_files(
    copy_patterns: Iterable[str],
    src_dir: Path,
    dest_dir: Path,
    paths_per_pair: int,
) -> None:
    for pattern in copy_patterns:
        formatted = pattern.replace("{paths}", str(paths_per_pair))
        src = src_dir / formatted
        if not src.exists():
            continue
        dst = dest_dir / formatted
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_filename = args.train_file_pattern.format(paths=args.paths_per_pair)
    src_train = args.src_dir / train_filename
    if not src_train.exists():
        raise FileNotFoundError(f"Training file not found: {src_train}")

    args.dest_dir.mkdir(parents=True, exist_ok=True)

    original_lines = load_train_lines(src_train)

    augmented_lines: List[str] = []
    prefix_count = 0
    suffix_count = 0

    for line in original_lines:
        parts = line.split()
        if len(parts) <= 2:
            continue  # malformed
        src_token, dst_token = parts[0], parts[1]
        path_nodes = parts[2:]

        if args.prefix_samples > 0:
            prefix_crops = generate_prefix_crops(
                path_nodes=path_nodes,
                prefix_samples=args.prefix_samples,
                min_len=args.min_prefix_len,
                max_len=args.max_prefix_len,
                rng=rng,
            )
            for crop in prefix_crops:
                augmented_lines.append(" ".join([src_token, dst_token] + crop))
            prefix_count += len(prefix_crops)

        if args.suffix_samples > 0:
            suffix_crops = generate_suffix_crops(
                path_nodes=path_nodes,
                suffix_samples=args.suffix_samples,
                min_len=args.min_suffix_len,
                max_len=args.max_suffix_len,
                rng=rng,
            )
            for crop in suffix_crops:
                augmented_lines.append(" ".join([src_token, dst_token] + crop))
            suffix_count += len(suffix_crops)

    final_lines: List[str] = []
    kept_original = args.keep_original == 1
    if kept_original:
        final_lines.extend(original_lines)
    final_lines.extend(augmented_lines)

    if args.shuffle:
        rng.shuffle(final_lines)

    dest_train = args.dest_dir / train_filename
    dest_train.parent.mkdir(parents=True, exist_ok=True)
    with dest_train.open("w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")

    copy_side_files(args.copy_patterns, args.src_dir, args.dest_dir, args.paths_per_pair)

    summary = {
        "source_dir": str(args.src_dir.resolve()),
        "dest_dir": str(args.dest_dir.resolve()),
        "train_file": train_filename,
        "seed": args.seed,
        "keep_original": kept_original,
        "counts": {
            "original_lines": len(original_lines),
            "prefix_crops": prefix_count,
            "suffix_crops": suffix_count,
            "augmented_lines": len(augmented_lines),
            "final_lines": len(final_lines),
        },
        "parameters": {
            "prefix_samples": args.prefix_samples,
            "suffix_samples": args.suffix_samples,
            "min_prefix_len": args.min_prefix_len,
            "max_prefix_len": args.max_prefix_len,
            "min_suffix_len": args.min_suffix_len,
            "max_suffix_len": args.max_suffix_len,
            "shuffle": args.shuffle,
        },
        "copy_patterns": args.copy_patterns,
    }

    summary_path = args.dest_dir / "augmentation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Partial-path augmentation complete.")
    print(f"Source dataset: {args.src_dir}")
    print(f"Destination   : {args.dest_dir}")
    print(f"Train file    : {train_filename}")
    print("- Counts")
    print(f"  Original lines : {len(original_lines):>10d}")
    print(f"  Prefix crops   : {prefix_count:>10d}")
    print(f"  Suffix crops   : {suffix_count:>10d}")
    print(f"  Augmented lines: {len(augmented_lines):>10d}")
    print(f"  Final lines    : {len(final_lines):>10d}")
    print("- Flags")
    print(f"  Keep original? : {'yes' if kept_original else 'no'}")
    print(f"  Shuffle output?: {'yes' if args.shuffle else 'no'}")
    print("=" * 70)


if __name__ == "__main__":
    main()