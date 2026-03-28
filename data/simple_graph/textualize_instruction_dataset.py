#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


TEMPLATES: Dict[str, List[str]] = {
    "plain_v1": [
        "instruction", "find", "a", "valid", "path", "from", "node", "{s}",
        "to", "node", "{t}",
        "output", "only", "the", "final", "path",
        "answer", "{s}",
    ],
    "stepwise_v1": [
        "instruction", "find", "a", "path", "from", "node", "{s}",
        "to", "node", "{t}",
        "step", "by", "step",
        "through", "intermediate", "nodes",
        "output", "only", "the", "final", "path",
        "answer", "{s}",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert numeric GraphA dataset into instruction-style text.")
    p.add_argument("--src-dir", type=Path, required=True, help="Source dataset dir with train_K.txt and test.txt")
    p.add_argument("--dest-dir", type=Path, required=True, help="Destination dir for textualized dataset")
    p.add_argument("--train-paths-per-pair", type=int, default=20, help="Used to locate train_{K}.txt")
    p.add_argument("--template", type=str, default="stepwise_v1", choices=sorted(TEMPLATES.keys()))
    p.add_argument(
        "--copy-patterns",
        nargs="*",
        default=[
            "composition_graph.graphml",
            "stage_info.pkl",
            "dataset_summary.json",
            "stage_skip_filter_summary.json",
            "pair_report.csv",
        ],
        help="Side files to copy if present.",
    )
    return p.parse_args()


def format_prompt(template_tokens: List[str], source: str, target: str) -> List[str]:
    out: List[str] = []
    for tok in template_tokens:
        if tok == "{s}":
            out.append(source)
        elif tok == "{t}":
            out.append(target)
        else:
            out.append(tok)
    return out


def convert_numeric_line(line: str, template_tokens: List[str]) -> str:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Each line must be: src tgt path..., got: {line}")

    source = parts[0]
    target = parts[1]
    path = parts[2:]

    if path[0] != source:
        raise ValueError(f"Path must start with source. line={line}")
    if path[-1] != target:
        raise ValueError(f"Path must end with target. line={line}")

    prompt = format_prompt(template_tokens, source, target)

    # prompt already ends with source after "answer {s}"
    continuation = path[1:]
    return " ".join(prompt + continuation)


def convert_file(src_file: Path, dest_file: Path, template_tokens: List[str]) -> int:
    count = 0
    with open(src_file, "r", encoding="utf-8") as fin, open(dest_file, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            fout.write(convert_numeric_line(line, template_tokens) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    src_train = args.src_dir / f"train_{args.train_paths_per_pair}.txt"
    src_test = args.src_dir / "test.txt"

    if not src_train.exists():
        raise FileNotFoundError(f"Missing training file: {src_train}")
    if not src_test.exists():
        raise FileNotFoundError(f"Missing test file: {src_test}")

    template_tokens = TEMPLATES[args.template]

    # 1) Copy numeric raw files for exact evaluation / RL pair loading
    dest_train_raw = args.dest_dir / f"train_raw_{args.train_paths_per_pair}.txt"
    dest_test_raw = args.dest_dir / "test_raw.txt"
    shutil.copy2(src_train, dest_train_raw)
    shutil.copy2(src_test, dest_test_raw)

    # 2) Write textualized train/test to standard names
    dest_train = args.dest_dir / f"train_{args.train_paths_per_pair}.txt"
    dest_test = args.dest_dir / "test.txt"
    n_train = convert_file(src_train, dest_train, template_tokens)
    n_test = convert_file(src_test, dest_test, template_tokens)

    # 3) Copy side files
    for name in args.copy_patterns:
        src = args.src_dir / name
        if src.exists():
            dst = args.dest_dir / name
            if src.is_file():
                shutil.copy2(src, dst)

    # 4) Save template metadata
    meta = {
        "template_name": args.template,
        "prompt_template_tokens": template_tokens,
        "notes": (
            "Prompt already ends with 'answer {s}'. "
            "The model should generate the remaining path continuation only."
        ),
        "raw_train_file": dest_train_raw.name,
        "raw_test_file": dest_test_raw.name,
    }
    with open(args.dest_dir / "instruction_template.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # print one example
    with open(dest_train, "r", encoding="utf-8") as f:
        first_example = f.readline().strip()

    print("=" * 70)
    print(f"Textualized dataset written to: {args.dest_dir}")
    print(f"  train text : {dest_train} ({n_train} lines)")
    print(f"  test text  : {dest_test} ({n_test} lines)")
    print(f"  train raw  : {dest_train_raw}")
    print(f"  test raw   : {dest_test_raw}")
    print(f"  template   : {args.template}")
    print("Example textualized line:")
    print(first_example)
    print("=" * 70)


if __name__ == "__main__":
    main()