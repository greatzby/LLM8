#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append every direct edge of a GraphA DAG dataset into train_{K}.txt as length-2 paths.

Features
--------
- Supports copying the entire dataset directory to a new location before augmentation.
- Deduplicates edges already present in the training text.
- Can shuffle newly-added edges for randomness.
- Writes a JSON summary describing the augmentation.

Typical workflow
----------------
1. Create the graph and base dataset (train_{K}.txt) with your existing pipeline.
2. Run make_p13_variant.py to obtain the desired P13 split directory.
3. Run this script on the P13 directory to add direct edges.
4. Finally run prepare_compositionnew.py to produce .bin files.

Dependencies
------------
- Python 3.10+
- networkx

Author
------
GPT-5 Codex (OpenAI)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append every direct edge as a two-token path into train_{K}.txt."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory produced by make_p13_variant.py (contains train_{K}.txt).",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Optional destination directory. "
             "If omitted, defaults to <source>_with_direct_edges. "
             "Ignored when --inplace is set.",
    )
    parser.add_argument(
        "--train-paths-per-pair",
        type=int,
        default=20,
        help="Used to locate train_{K}.txt (default: 20).",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Optional explicit path to composition_graph.graphml. "
             "Defaults to <work-dir>/composition_graph.graphml.",
    )
    parser.add_argument(
        "--shuffle-added",
        action="store_true",
        help="Shuffle the newly-added direct edges before appending.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for shuffling (default: 2025).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify the source directory in place (no copy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing --dest-dir.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When used with --inplace, create a train backup "
             "as train_{K}.txt.bak timestamped before overriding.",
    )
    return parser.parse_args()


def resolve_work_dir(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    train_name = f"train_{args.train_paths_per_pair}.txt"
    source_train = source_dir / train_name
    if not source_train.exists():
        raise FileNotFoundError(f"Training file not found: {source_train}")

    if args.inplace:
        if args.dest_dir:
            raise ValueError("Do not supply --dest-dir when using --inplace.")
        work_dir = source_dir
        mode = "inplace"
    else:
        if args.dest_dir:
            work_dir = args.dest_dir.resolve()
        else:
            work_dir = source_dir.with_name(source_dir.name + "_with_direct_edges")

        if work_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Destination directory already exists: {work_dir}\n"
                    "Use --overwrite to allow copying into it (contents may be replaced)."
                )
        if work_dir != source_dir:
            shutil.copytree(source_dir, work_dir, dirs_exist_ok=True)
        mode = "copied"

    work_train = work_dir / train_name
    return work_dir, work_train, mode


def load_training_lines(train_path: Path) -> List[str]:
    lines: List[str] = []
    with train_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                lines.append(stripped)
    return lines


def load_graph_edges(graph_path: Path) -> List[Tuple[str, str]]:
    if not graph_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graph_path}")

    graph = nx.read_graphml(graph_path)
    edges = [(str(u), str(v)) for u, v in graph.edges()]
    return edges


def edge_line(src: str, dst: str) -> str:
    return f"{src} {dst}"


def maybe_backup_train(train_path: Path) -> Path | None:
    if not train_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = train_path.with_suffix(train_path.suffix + f".bak_{timestamp}")
    shutil.copy2(train_path, backup_path)
    return backup_path


def sort_edge_lines(lines: Iterable[str]) -> List[str]:
    def key_fn(line: str):
        tokens = line.split()
        numeric_tokens = []
        for tok in tokens:
            try:
                numeric_tokens.append(int(tok))
            except ValueError:
                numeric_tokens.append(tok)
        return numeric_tokens

    return sorted(lines, key=key_fn)


def write_train_lines(train_path: Path, lines: Iterable[str]) -> None:
    with train_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_summary(
    work_dir: Path,
    summary: dict,
    filename: str = "direct_edge_augmentation_summary.json",
) -> None:
    summary_path = work_dir / filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    work_dir, train_path, mode = resolve_work_dir(args)

    if args.inplace and args.backup:
        backup_path = maybe_backup_train(train_path)
        if backup_path:
            print(f"[info] train backup created: {backup_path}")
        else:
            print("[warn] train backup requested but train file missing; skipped.")

    lines_before = load_training_lines(train_path)
    line_set = set(lines_before)

    graph_path = args.graph_path.resolve() if args.graph_path else (work_dir / "composition_graph.graphml")
    edges = load_graph_edges(graph_path)

    direct_lines = [edge_line(src, dst) for src, dst in edges]
    unique_direct_lines = [line for line in direct_lines if line not in line_set]

    duplicates = len(direct_lines) - len(unique_direct_lines)
    if args.shuffle_added and unique_direct_lines:
        rng = random.Random(args.seed)
        rng.shuffle(unique_direct_lines)
    else:
        unique_direct_lines = sort_edge_lines(unique_direct_lines)

    final_lines = lines_before + unique_direct_lines
    write_train_lines(train_path, final_lines)

    summary = {
        "operation": "add_direct_edges",
        "mode": mode,
        "source_dir": str(args.source_dir.resolve()),
        "work_dir": str(work_dir),
        "train_file": train_path.name,
        "graph_path": str(graph_path),
        "train_samples_before": len(lines_before),
        "train_samples_after": len(final_lines),
        "edges_in_graph": len(direct_lines),
        "edges_already_present": duplicates,
        "edges_added": len(unique_direct_lines),
        "shuffle_added": args.shuffle_added,
        "seed": args.seed if args.shuffle_added else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_summary(work_dir, summary)

    print("=" * 70)
    print("Direct edge augmentation completed.")
    print(f"  Source dir         : {summary['source_dir']}")
    print(f"  Working dir        : {summary['work_dir']}")
    print(f"  Train file         : {train_path}")
    print(f"  Graph path         : {graph_path}")
    print("- Counts")
    print(f"  Train lines before : {summary['train_samples_before']}")
    print(f"  Train lines after  : {summary['train_samples_after']}")
    print(f"  Total graph edges  : {summary['edges_in_graph']}")
    print(f"  Already present    : {summary['edges_already_present']}")
    print(f"  Newly added edges  : {summary['edges_added']}")
    print("=" * 70)


if __name__ == "__main__":
    main()