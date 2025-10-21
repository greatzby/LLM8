#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier-3 (ALPINE-style) dataset generator for GraphA experiments.

Usage example:
    python generate_alpine_from_graph.py \
        --input_graph data/graphs/graphA_pg020_nps30_seed42/composition_graph.graphml \
        --stage_info data/graphs/graphA_pg020_nps30_seed42/stage_info.pkl \
        --output_dir data/datasets/graphA_pg020_tier3 \
        --train_paths_per_pair 20 \
        --eval_paths_per_pair 1 \
        --train_ratio 0.85 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

Node = str
Pair = Tuple[Node, Node]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ALPINE-style dataset (Tier-3).")
    parser.add_argument("--input_graph", type=str, required=True,
                        help="Path to composition_graph.graphml.")
    parser.add_argument("--stage_info", type=str, required=True,
                        help="Path to stage_info.pkl produced by graph generator.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store generated dataset.")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="Number of random paths per training pair.")
    parser.add_argument("--eval_paths_per_pair", type=int, default=1,
                        help="Number of paths per evaluation pair.")
    parser.add_argument("--train_ratio", type=float, default=0.85,
                        help="Train ratio for non-direct-edge pairs (0~1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--max_path_attempts", type=int, default=50,
                        help="Max retries when failing to sample a random path.")
    parser.add_argument("--verbose_examples", type=int, default=0,
                        help="Number of sample pairs to print for sanity check.")
    return parser.parse_args()


def load_graph(graph_path: Path) -> nx.DiGraph:
    G = nx.read_graphml(graph_path)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph must be a DAG for Tier-3 generation.")
    return G


def load_stage_info(stage_info_path: Path) -> dict:
    with open(stage_info_path, "rb") as f:
        return pickle.load(f)


def precompute_reachability(G: nx.DiGraph) -> Dict[Node, set[Node]]:
    """
    For each target node compute the set of ancestor nodes (including itself).
    """
    reachability: Dict[Node, set[Node]] = {}
    for node in tqdm(G.nodes, desc="Pre-computing reachability"):
        ancestors = nx.ancestors(G, node)
        ancestors.add(node)
        reachability[node] = ancestors
    return reachability


def generate_random_path(
    G: nx.DiGraph,
    source: Node,
    target: Node,
    reachability_cache: Dict[Node, set[Node]],
    max_attempts: int = 50,
) -> Optional[List[int]]:
    """Sample a random path (Tier-3 style) from source to target."""
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        path: List[Node] = [source]
        current = source
        visited = {source}

        while current != target:
            successors = list(G.successors(current))
            valid = [n for n in successors if n in reachability_cache[target] and n not in visited]
            if not valid:
                # allow re-visiting if DAG but path forced?
                valid = [n for n in successors if n in reachability_cache[target]]
            if not valid:
                break
            current = random.choice(valid)
            path.append(current)
            if len(path) > G.number_of_nodes():
                break  # safety guard

        if path[-1] == target:
            return [int(node) for node in path]

    return None


def classify_pair(
    src: int,
    dst: int,
    stages: Sequence[Sequence[int]],
) -> Optional[str]:
    S1, S2, S3 = stages[:3]
    if src in S1 and dst in S2:
        return "S1->S2"
    if src in S2 and dst in S3:
        return "S2->S3"
    if src in S1 and dst in S3:
        return "S1->S3"
    return None


def stratified_split_pairs(
    pairs_by_type: Dict[str, List[Pair]],
    G: nx.DiGraph,
    train_ratio: float,
) -> tuple[List[Pair], List[Pair]]:
    train_pairs: List[Pair] = []
    test_pairs: List[Pair] = []
    for path_type, pair_list in pairs_by_type.items():
        random.shuffle(pair_list)
        direct_pairs = [pair for pair in pair_list if G.has_edge(*pair)]
        non_direct_pairs = [pair for pair in pair_list if not G.has_edge(*pair)]

        train_pairs.extend(direct_pairs)  # enforce ALPINE rule

        if non_direct_pairs:
            cutoff = int(len(non_direct_pairs) * train_ratio)
            train_pairs.extend(non_direct_pairs[:cutoff])
            test_pairs.extend(non_direct_pairs[cutoff:])

    return train_pairs, test_pairs


def create_dataset(
    G: nx.DiGraph,
    stages: Sequence[Sequence[int]],
    reachability_cache: Dict[Node, set[Node]],
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    train_paths_per_pair: int,
    eval_paths_per_pair: int,
    max_path_attempts: int,
) -> tuple[List[List[int]], List[List[int]], Dict[str, Dict[str, int]]]:
    train_samples: List[List[int]] = []
    test_samples: List[List[int]] = []

    sample_counts_by_split: Dict[str, defaultdict[str, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
    }

    pair_type_cache: Dict[Pair, str] = {}

    def get_pair_type(pair: Pair) -> str:
        if pair not in pair_type_cache:
            src_str, dst_str = pair
            pair_type = classify_pair(int(src_str), int(dst_str), stages)
            pair_type_cache[pair] = pair_type or "Other"
        return pair_type_cache[pair]

    def record_sample(
        container: List[List[int]],
        pair: Pair,
        path: Sequence[int],
        split: str,
    ) -> None:
        src_int = int(pair[0])
        dst_int = int(pair[1])
        path_int = [int(node) for node in path]
        sample = [src_int, dst_int] + path_int
        container.append(sample)

        pair_type = get_pair_type(pair)
        sample_counts_by_split[split][pair_type] += 1
        sample_counts_by_split[split]["__total__"] += 1

    # Training samples
    for source, target in tqdm(train_pairs, desc="Generating training samples"):
        pair = (source, target)
        if G.has_edge(source, target):
            direct_path = [int(source), int(target)]
            record_sample(train_samples, pair, direct_path, "train")

        for _ in range(train_paths_per_pair):
            path = generate_random_path(
                G, source, target, reachability_cache, max_attempts=max_path_attempts
            )
            if path is not None:
                record_sample(train_samples, pair, path, "train")

    # Testing samples
    for source, target in tqdm(test_pairs, desc="Generating eval samples"):
        pair = (source, target)
        for _ in range(eval_paths_per_pair):
            path = generate_random_path(
                G, source, target, reachability_cache, max_attempts=max_path_attempts
            )
            if path is not None:
                record_sample(test_samples, pair, path, "test")

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # Convert defaultdicts to plain dict for serialization
    sample_counts_dict = {split: dict(counts) for split, counts in sample_counts_by_split.items()}
    return train_samples, test_samples, sample_counts_dict


def write_dataset(lines: Iterable[List[int]], file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(" ".join(str(x) for x in line))
            f.write("\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    graph_path = Path(args.input_graph).resolve()
    stage_info_path = Path(args.stage_info).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph(graph_path)
    stage_info = load_stage_info(stage_info_path)
    stages: List[List[int]] = stage_info["stages"]

    reachability_cache = precompute_reachability(G)

    nodes = list(G.nodes)
    pairs_by_type: Dict[str, List[Pair]] = defaultdict(list)

    for src in tqdm(nodes, desc="Collecting reachable pairs"):
        for dst in nodes:
            if src == dst:
                continue
            if src in reachability_cache[dst]:
                src_int, dst_int = int(src), int(dst)
                pair_type = classify_pair(src_int, dst_int, stages)
                if pair_type:
                    pairs_by_type[pair_type].append((src, dst))

    train_pairs, test_pairs = stratified_split_pairs(pairs_by_type, G, args.train_ratio)

    print("=" * 70)
    print("Reachable pair statistics (before path sampling):")
    for path_type, pair_list in pairs_by_type.items():
        train_count = sum(1 for pair in train_pairs if pair in pair_list)
        test_count = sum(1 for pair in test_pairs if pair in pair_list)
        print(f"  {path_type}: total={len(pair_list)}, train={train_count}, test={test_count}")
    print("=" * 70)

    train_samples, test_samples, sample_counts = create_dataset(
        G=G,
        stages=stages,
        reachability_cache=reachability_cache,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        train_paths_per_pair=args.train_paths_per_pair,
        eval_paths_per_pair=args.eval_paths_per_pair,
        max_path_attempts=args.max_path_attempts,
    )

    train_file = output_dir / f"train_{args.train_paths_per_pair}.txt"
    test_file = output_dir / "test.txt"

    write_dataset(train_samples, train_file)
    write_dataset(test_samples, test_file)

    # Copy graph & stage info for downstream scripts
    nx.write_graphml(G, output_dir / "composition_graph.graphml")
    with open(output_dir / "stage_info.pkl", "wb") as f:
        pickle.dump(stage_info, f)

    summary = {
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_paths_per_pair": args.train_paths_per_pair,
        "eval_paths_per_pair": args.eval_paths_per_pair,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "pair_counts": {k: len(v) for k, v in pairs_by_type.items()},
        "sample_counts": {
            "train": sample_counts["train"],
            "test": sample_counts["test"],
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
        },
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Dataset saved to: {output_dir}")
    print(f"  Train file: {train_file} (samples={len(train_samples)})")
    print(f"  Test file : {test_file} (samples={len(test_samples)})")
    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  - {key}:")
            for sub_key, sub_val in value.items():
                print(f"      * {sub_key}: {sub_val}")
        else:
            print(f"  - {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()