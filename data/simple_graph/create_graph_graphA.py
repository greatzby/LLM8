#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphA generator for multi-stage DAG experiments.

Usage example:
    python data/simple_graph/create_graph_graphA.py \
        --nodes_per_stage 30 \
        --p_global 0.20 \
        --seed 42 \
        --experiment_name graphA \
        --output_root data/graphs

This will create a folder like:
    data/graphs/graphA_pg020_nps30_seed42/
containing:
    - composition_graph.graphml
    - stage_info.pkl
    - metadata.json
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np


def build_graph_a(
    nodes_per_stage: int,
    num_stages: int,
    p_global: float,
    p_intra: float,
    allow_stage_skip: bool,
) -> tuple[nx.DiGraph, list[list[int]]]:
    """Construct a layered DAG following GraphA rules."""
    total_nodes = nodes_per_stage * num_stages
    G = nx.DiGraph()
    G.add_nodes_from(str(i) for i in range(total_nodes))

    stages: list[list[int]] = []
    for stage_idx in range(num_stages):
        start = stage_idx * nodes_per_stage
        stop = (stage_idx + 1) * nodes_per_stage
        stages.append(list(range(start, stop)))

    # Intra-stage edges (i < j to keep DAG)
    for stage_nodes in stages:
        for i_idx, src in enumerate(stage_nodes[:-1]):
            for dst in stage_nodes[i_idx + 1:]:
                if random.random() < p_intra:
                    G.add_edge(str(src), str(dst))

    # Inter-stage edges
    for src_stage_idx in range(num_stages - 1):
        for dst_stage_idx in range(src_stage_idx + 1, num_stages):
            if not allow_stage_skip and dst_stage_idx != src_stage_idx + 1:
                continue
            src_nodes = stages[src_stage_idx]
            dst_nodes = stages[dst_stage_idx]
            for src in src_nodes:
                for dst in dst_nodes:
                    if random.random() < p_global:
                        G.add_edge(str(src), str(dst))

    return G, stages


def summarize_graph(G: nx.DiGraph, stages: list[list[int]]) -> dict:
    """Compute simple statistics for logging."""
    summary: dict[str, float | int] = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }
    for idx, nodes in enumerate(stages, start=1):
        summary[f"stage_{idx}_size"] = len(nodes)

    # Reachable pairs per bucket
    categories = {"S1->S2": 0, "S2->S3": 0, "S1->S3": 0}
    S1, S2, S3 = stages
    for src in S1:
        for dst in S2:
            if nx.has_path(G, str(src), str(dst)):
                categories["S1->S2"] += 1
        for dst in S3:
            if nx.has_path(G, str(src), str(dst)):
                categories["S1->S3"] += 1
    for src in S2:
        for dst in S3:
            if nx.has_path(G, str(src), str(dst)):
                categories["S2->S3"] += 1

    summary.update(categories)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GraphA DAG.")
    parser.add_argument("--nodes_per_stage", type=int, default=30,
                        help="Number of nodes per stage (default: 30).")
    parser.add_argument("--num_stages", type=int, default=3,
                        help="Total stages (default: 3).")
    parser.add_argument("--p_global", type=float, required=True,
                        help="Edge probability for cross-stage edges.")
    parser.add_argument("--p_intra", type=float, default=None,
                        help="Edge probability within stage (defaults to p_global/2).")
    parser.add_argument("--allow_stage_skip", action="store_true",
                        help="Allow edges beyond immediate next stage.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--experiment_name", type=str, default="graphA",
                        help="Experiment prefix for folder naming.")
    parser.add_argument("--output_root", type=str, default="data/graphs",
                        help="Root directory where the graph folder will be created.")
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    suffix = f"pg{int(args.p_global * 100):03d}_nps{args.nodes_per_stage}_seed{args.seed}"
    folder_name = f"{args.experiment_name}_{suffix}"
    out_dir = Path(args.output_root).resolve() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    if args.p_intra is None:
        args.p_intra = max(min(args.p_global / 2.0, 1.0), 0.0)

    random.seed(args.seed)
    np.random.seed(args.seed)

    G, stages = build_graph_a(
        nodes_per_stage=args.nodes_per_stage,
        num_stages=args.num_stages,
        p_global=args.p_global,
        p_intra=args.p_intra,
        allow_stage_skip=args.allow_stage_skip,
    )

    out_dir = make_output_dir(args)

    graph_path = out_dir / "composition_graph.graphml"
    nx.write_graphml(G, graph_path)

    stage_info = {
        "stages": stages,
        "nodes_per_stage": args.nodes_per_stage,
        "num_stages": args.num_stages,
        "allow_stage_skip": args.allow_stage_skip,
    }
    with open(out_dir / "stage_info.pkl", "wb") as f:
        pickle.dump(stage_info, f)

    summary = summarize_graph(G, stages)
    summary.update({
        "p_global": args.p_global,
        "p_intra": args.p_intra,
        "allow_stage_skip": args.allow_stage_skip,
        "seed": args.seed,
    })

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"GraphA DAG saved to: {out_dir}")
    print(f"GraphML: {graph_path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()