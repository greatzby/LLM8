#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create control splits with specific P13 ratio from GraphA tier-3 dataset.

Example usage:

  python make_p13_variant.py \
      --src-dir data/datasets/graphA_pg030_tier3 \
      --dest-dir data/datasets/graphA_pg030_tier3_P13_0 \
      --target-ratio 0.0 \
      --paths-per-pair 20

  python make_p13_variant.py \
      --src-dir data/datasets/graphA_pg030_tier3 \
      --dest-dir data/datasets/graphA_pg030_tier3_P13_20 \
      --target-ratio 0.2 \
      --paths-per-pair 20 \
      --seed 1234
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build control dataset with desired P13 ratio.")
    parser.add_argument("--src-dir", type=Path, required=True, help="Original dataset directory.")
    parser.add_argument("--dest-dir", type=Path, required=True, help="Destination directory.")
    parser.add_argument("--paths-per-pair", type=int, default=20,
                        help="Used to locate train_{K}.txt.")
    parser.add_argument("--target-ratio", type=float, required=True,
                        help="Fraction of (s,t) pairs with s∈S1, t∈S3 after subsampling.")
    parser.add_argument("--stage-info-name", type=str, default="stage_info.pkl",
                        help="Stage info file name inside src-dir.")
    parser.add_argument("--train-file-pattern", type=str, default="train_{paths}.txt",
                        help="Format string for the training file.")
    parser.add_argument("--source-index", type=int, default=0,
                        help="Token index for source node (supports negative index).")
    parser.add_argument("--target-index", type=int, default=-1,
                        help="Token index for target node (supports negative index).")
    parser.add_argument("--s1-nodes", type=str, default=None,
                        help="Comma-separated list of S1 node ids (overrides auto-detect).")
    parser.add_argument("--s3-nodes", type=str, default=None,
                        help="Comma-separated list of S3 node ids (overrides auto-detect).")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for subsampling.")
    parser.add_argument(
        "--copy-patterns",
        nargs="*",
        default=[
            "composition_graph.graphml",
            "dataset_summary.json",
            "meta.pkl",
            "stage_info.pkl",
            "test.txt",
            "val.bin",
            "train_{paths}.bin",
        ],
        help="Files (or directories) to copy from src to dest if they exist.",
    )
    return parser.parse_args()


def normalize_stage_nodes(nodes: Sequence) -> List[str]:
    """Convert stage node identifiers to strings (match path tokens)."""
    result = []
    for node in nodes:
        if isinstance(node, (int, float)):
            result.append(str(int(node)))
        else:
            result.append(str(node))
    return result


def infer_stage_sets(stage_info: dict) -> Dict[str, Sequence[str]]:
    """Try best-effort extraction of tier node sets."""

    # --- Case 1: stage_info has "stages" list (GraphA generator output) ---
    stages = stage_info.get("stages")
    if isinstance(stages, (list, tuple)) and len(stages) >= 3:
        return {
            "s1": normalize_stage_nodes(stages[0]),
            "s3": normalize_stage_nodes(stages[2]),
        }

    # --- Case 2: generic dict-of-dicts (tier1/tier2/etc.) ---
    candidates = []
    for key, value in stage_info.items():
        if isinstance(value, dict) and len(value) >= 3:
            canonical = {}
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (list, tuple, set)):
                    canonical[str(sub_key).lower()] = normalize_stage_nodes(sub_val)
            if len(canonical) >= 3:
                candidates.append(canonical)

    if not candidates:
        raise ValueError(
            "Cannot automatically infer tier node sets. "
            "Please supply --s1-nodes/--s3-nodes manually."
        )

    tiers = candidates[0]
    aliases = {
        "s1": ["s1", "tier1", "stage1", "cluster1"],
        "s3": ["s3", "tier3", "stage3", "cluster3"],
    }

    resolved = {}
    for name, keys in aliases.items():
        for key in keys:
            if key in tiers:
                resolved[name] = tiers[key]
                break

    if "s1" not in resolved or "s3" not in resolved:
        raise ValueError(
            f"Inferred tiers but missing S1/S3 keys. Found keys: {list(tiers.keys())}. "
            "Use --s1-nodes/--s3-nodes."
        )

    return resolved


def load_stage_sets(args: argparse.Namespace) -> Tuple[set, set]:
    if args.s1_nodes and args.s3_nodes:
        s1 = {token.strip() for token in args.s1_nodes.split(",") if token.strip()}
        s3 = {token.strip() for token in args.s3_nodes.split(",") if token.strip()}
        if not s1 or not s3:
            raise ValueError("S1 or S3 list is empty after parsing CLI arguments.")
        return s1, s3

    stage_path = args.src_dir / args.stage_info_name
    if not stage_path.exists():
        raise FileNotFoundError(f"Stage info file not found: {stage_path}")

    with open(stage_path, "rb") as f:
        stage_info = pickle.load(f)

    if not isinstance(stage_info, dict):
        raise TypeError(
            f"stage_info.pkl must contain a dict, but got {type(stage_info).__name__}"
        )

    tiers = infer_stage_sets(stage_info)
    s1 = set(tiers["s1"])
    s3 = set(tiers["s3"])
    if not s1 or not s3:
        raise ValueError("Parsed S1 or S3 node sets are empty.")
    return s1, s3


def canonical_token(token: str) -> str:
    return token.strip()


def resolve_index(tokens: List[str], index: int) -> str:
    if index >= 0:
        if index >= len(tokens):
            raise IndexError(f"Token index {index} out of range for tokens {tokens}")
        return tokens[index]
    resolved = len(tokens) + index
    if resolved < 0:
        raise IndexError(f"Token index {index} out of range for tokens {tokens}")
    return tokens[resolved]


def split_pairs(
    lines: List[str],
    source_idx: int,
    target_idx: int,
) -> Dict[Tuple[str, str], List[str]]:
    bucket: Dict[Tuple[str, str], List[str]] = {}
    for line in lines:
        parts = [canonical_token(tok) for tok in line.split()]
        if not parts:
            continue
        src = resolve_index(parts, source_idx)
        tgt = resolve_index(parts, target_idx)
        key = (src, tgt)
        bucket.setdefault(key, []).append(line)
    return bucket


def compute_subset_sizes(
    n_p13: int,
    n_p0: int,
    target_ratio: float,
) -> Tuple[int, int]:
    if target_ratio < 0 or target_ratio > 1:
        raise ValueError("target_ratio must be in [0, 1].")

    if n_p13 + n_p0 == 0:
        return 0, 0

    if target_ratio == 0:
        return 0, n_p0
    if target_ratio == 1:
        return n_p13, 0

    orig_ratio = n_p13 / (n_p13 + n_p0)

    if target_ratio <= orig_ratio:
        keep_p0 = n_p0
        desired_p13 = target_ratio * keep_p0 / (1 - target_ratio)
        keep_p13 = min(n_p13, int(round(desired_p13)))
        return keep_p13, keep_p0

    keep_p13 = n_p13
    desired_p0 = keep_p13 * (1 - target_ratio) / target_ratio
    keep_p0 = min(n_p0, int(round(desired_p0)))
    return keep_p13, keep_p0


def select_pairs(
    p13_pairs: List[Tuple[str, str]],
    p0_pairs: List[Tuple[str, str]],
    keep_p13: int,
    keep_p0: int,
    seed: int,
) -> Tuple[set, set]:
    rng = random.Random(seed)
    if keep_p13 > len(p13_pairs):
        keep_p13 = len(p13_pairs)
    if keep_p0 > len(p0_pairs):
        keep_p0 = len(p0_pairs)
    selected_p13 = set(rng.sample(p13_pairs, keep_p13)) if keep_p13 else set()
    selected_p0 = set(rng.sample(p0_pairs, keep_p0)) if keep_p0 else set()
    return selected_p13, selected_p0


def copy_side_files(args: argparse.Namespace) -> None:
    for pattern in args.copy_patterns:
        formatted = pattern.replace("{paths}", str(args.paths_per_pair))
        src = args.src_dir / formatted
        if not src.exists():
            continue
        dst = args.dest_dir / formatted
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    train_name = args.train_file_pattern.format(paths=args.paths_per_pair)
    src_train = args.src_dir / train_name
    if not src_train.exists():
        raise FileNotFoundError(f"Training file not found: {src_train}")

    with open(src_train, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f if line.strip()]

    s1_nodes, s3_nodes = load_stage_sets(args)

    pair_to_lines = split_pairs(raw_lines, args.source_index, args.target_index)
    p13_pairs, p0_pairs = [], []
    for pair in pair_to_lines:
        src, tgt = pair
        if src in s1_nodes and tgt in s3_nodes:
            p13_pairs.append(pair)
        else:
            p0_pairs.append(pair)

    n_p13 = len(p13_pairs)
    n_p0 = len(p0_pairs)
    if n_p13 == 0 and args.target_ratio > 0:
        raise ValueError("No P13 pairs available in source data; cannot reach target ratio > 0.")

    keep_p13, keep_p0 = compute_subset_sizes(n_p13, n_p0, args.target_ratio)
    selected_p13, selected_p0 = select_pairs(p13_pairs, p0_pairs, keep_p13, keep_p0, args.seed)

    keep_pairs = selected_p13 | selected_p0
    kept_lines: List[str] = []
    for pair, lines in pair_to_lines.items():
        if pair in keep_pairs:
            kept_lines.extend(lines)

    rng = random.Random(args.seed)
    rng.shuffle(kept_lines)

    dest_train = args.dest_dir / train_name
    with open(dest_train, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    copy_side_files(args)

    summary = {
        "target_ratio": args.target_ratio,
        "actual_ratio": (
            len(selected_p13) / (len(selected_p13) + len(selected_p0))
            if (selected_p13 or selected_p0)
            else 0.0
        ),
        "total_pairs": len(selected_p13) + len(selected_p0),
        "kept_p13_pairs": len(selected_p13),
        "kept_p0_pairs": len(selected_p0),
        "paths_per_pair": args.paths_per_pair,
        "source_index": args.source_index,
        "target_index": args.target_index,
        "seed": args.seed,
    }
    with open(args.dest_dir / "p13_control_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Created control split in {args.dest_dir}")
    print(f"  target ratio     : {args.target_ratio:.3f}")
    print(f"  achieved ratio   : {summary['actual_ratio']:.3f}")
    print(f"  kept P13 pairs   : {summary['kept_p13_pairs']}")
    print(f"  kept P0  pairs   : {summary['kept_p0_pairs']}")
    print(f"  total kept pairs : {summary['total_pairs']}")
    print("=" * 70)


if __name__ == "__main__":
    main()