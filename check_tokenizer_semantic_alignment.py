#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def sha256_file(p: Path) -> Optional[str]:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def try_load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def get_tokenizer_dir(ckpt_dir: Path) -> Path:
    """
    Heuristic:
    - if ckpt_dir has tokenizer.json, use it
    - else try ckpt_dir/tokenizer
    - else return ckpt_dir (AutoTokenizer will try)
    """
    if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "tokenizer_config.json").exists():
        return ckpt_dir
    if (ckpt_dir / "tokenizer" / "tokenizer.json").exists() or (ckpt_dir / "tokenizer" / "tokenizer_config.json").exists():
        return ckpt_dir / "tokenizer"
    return ckpt_dir

def tok_fingerprint(tok_dir: Path) -> str:
    """
    Prefer hashing tokenizer.json (strongest).
    Fall back to hashing tokenizer_config.json or special_tokens_map.json if exists.
    """
    for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        h = sha256_file(tok_dir / name)
        if h:
            return f"{name}:sha256={h}"
    return "no_tokenizer_files_hashed"

def summarize_tokenizer(name: str, tok, tok_dir: Path) -> None:
    print(f"\n=== [{name}] tokenizer summary ===")
    print(f"[{name}] dir = {tok_dir}")
    print(f"[{name}] class = {tok.__class__.__name__}")
    print(f"[{name}] len(tokenizer) = {len(tok)}")
    print(f"[{name}] fingerprint = {tok_fingerprint(tok_dir)}")

    # Special tokens
    def safe_id(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    pad_id = safe_id(getattr(tok, "pad_token_id", None))
    eos_id = safe_id(getattr(tok, "eos_token_id", None))
    bos_id = safe_id(getattr(tok, "bos_token_id", None))
    unk_id = safe_id(getattr(tok, "unk_token_id", None))

    print(f"[{name}] pad_token = {repr(getattr(tok, 'pad_token', None))}  pad_id = {pad_id}")
    print(f"[{name}] eos_token = {repr(getattr(tok, 'eos_token', None))}  eos_id = {eos_id}")
    print(f"[{name}] bos_token = {repr(getattr(tok, 'bos_token', None))}  bos_id = {bos_id}")
    print(f"[{name}] unk_token = {repr(getattr(tok, 'unk_token', None))}  unk_id = {unk_id}")

    # Tokens at those ids (if in range)
    for label, _id in [("pad", pad_id), ("eos", eos_id), ("bos", bos_id), ("unk", unk_id)]:
        if _id is None:
            continue
        if 0 <= _id < len(tok):
            try:
                t = tok.convert_ids_to_tokens(_id)
            except Exception as e:
                t = f"<error convert_ids_to_tokens: {e}>"
            print(f"[{name}] id->{label} token string: id={_id} token={repr(t)}")
        else:
            print(f"[{name}] WARNING: {label}_id={_id} is OUT OF RANGE for len={len(tok)}")

def compare_encodings(t1, t2, texts: List[str], name1: str, name2: str) -> None:
    print("\n=== [encoding compare] ===")
    for s in texts:
        a = t1.encode(s, add_special_tokens=False)
        b = t2.encode(s, add_special_tokens=False)
        same = (a == b)
        print(f"\nTEXT: {repr(s)}")
        print(f"{name1} ids (first 80): {a[:80]}")
        print(f"{name2} ids (first 80): {b[:80]}")
        print(f"same_ids? {same}")
        if not same:
            # show token pieces for the first differing position (if any)
            m = min(len(a), len(b))
            diff_pos = None
            for i in range(m):
                if a[i] != b[i]:
                    diff_pos = i
                    break
            if diff_pos is None and len(a) != len(b):
                diff_pos = m
            print(f"first_diff_pos = {diff_pos}")
            if diff_pos is not None:
                if diff_pos < len(a):
                    print(f"{name1} piece @diff: id={a[diff_pos]} tok={repr(t1.convert_ids_to_tokens(a[diff_pos]))}")
                if diff_pos < len(b):
                    print(f"{name2} piece @diff: id={b[diff_pos]} tok={repr(t2.convert_ids_to_tokens(b[diff_pos]))}")

def compare_vocab_mappings(t1, t2, name1: str, name2: str, max_examples: int = 20) -> None:
    """
    Strong check: compare get_vocab() dicts (token_string -> id).
    This is the real "semantic alignment" check.
    """
    print("\n=== [vocab mapping compare] (token_string -> id) ===")
    v1 = t1.get_vocab()
    v2 = t2.get_vocab()

    print(f"{name1} vocab size = {len(v1)}")
    print(f"{name2} vocab size = {len(v2)}")

    # Fast equality
    equal = (v1 == v2)
    print(f"vocab_dict_equal? {equal}")

    # If not equal, show stats
    if not equal:
        keys1 = set(v1.keys())
        keys2 = set(v2.keys())
        only1 = sorted(list(keys1 - keys2))[:max_examples]
        only2 = sorted(list(keys2 - keys1))[:max_examples]
        inter = keys1 & keys2

        # tokens that exist in both but have different ids
        diff = []
        for k in inter:
            if v1[k] != v2[k]:
                diff.append((k, v1[k], v2[k]))
        diff.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)

        print(f"tokens only in {name1}: {len(keys1-keys2)} (show up to {max_examples})")
        for k in only1:
            print(f"  {repr(k)} -> {v1[k]}")
        print(f"tokens only in {name2}: {len(keys2-keys1)} (show up to {max_examples})")
        for k in only2:
            print(f"  {repr(k)} -> {v2[k]}")

        print(f"tokens in both but id differs: {len(diff)} (show up to {max_examples})")
        for k, i1, i2 in diff[:max_examples]:
            print(f"  {repr(k)}: {name1}={i1}  {name2}={i2}")

def sample_id_token_checks(t1, t2, name1: str, name2: str, seed: int = 0, n_random: int = 30) -> None:
    print("\n=== [id -> token_string spot check] ===")
    random.seed(seed)
    max_len = min(len(t1), len(t2))
    ids = list(range(0, min(30, max_len)))
    if max_len > 60:
        ids += list(range(max_len - 30, max_len))
    for _ in range(n_random):
        ids.append(random.randint(0, max_len - 1))
    ids = sorted(set(ids))

    mismatch = 0
    for i in ids:
        s1 = t1.convert_ids_to_tokens(i)
        s2 = t2.convert_ids_to_tokens(i)
        if s1 != s2:
            mismatch += 1
            print(f"ID {i}: {name1}={repr(s1)}  {name2}={repr(s2)}")
    print(f"mismatch_count = {mismatch} / checked {len(ids)} ids (range limited to min_len={max_len})")

def maybe_load_texts_from_jsonl(jsonl_path: Path, key: str, limit: int = 30) -> List[str]:
    if not jsonl_path.exists():
        return []
    out = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if key in obj and isinstance(obj[key], str):
                out.append(obj[key])
            if len(out) >= limit:
                break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B",
                    help="HF model name or local dir for the base tokenizer")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Your SFT/RL checkpoint dir that contains tokenizer.json (e.g. .../ckpt_800)")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    ap.add_argument("--use_fast", action="store_true", default=True)
    ap.add_argument("--no_use_fast", dest="use_fast", action="store_false")

    ap.add_argument("--dataset_jsonl", type=str, default=None,
                    help="Optional: a jsonl file to sample texts from")
    ap.add_argument("--text_key", type=str, default="prompt",
                    help="Key in json lines to extract text from")
    ap.add_argument("--sample_texts", type=int, default=20)

    ap.add_argument("--skip_vocab_dict_compare", action="store_true",
                    help="Skip full get_vocab() dict equality (slower but strongest check)")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    # Lazy imports so script can print help without transformers installed
    from transformers import AutoTokenizer, AutoConfig

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_tok_dir = get_tokenizer_dir(ckpt_dir)

    # Load tokenizers
    t_base = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
    )
    t_ckpt = AutoTokenizer.from_pretrained(
        str(ckpt_tok_dir),
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
    )

    # Summaries
    summarize_tokenizer("BASE", t_base, Path(args.base_model) if os.path.isdir(args.base_model) else Path("."))
    summarize_tokenizer("CKPT", t_ckpt, ckpt_tok_dir)

    # Base config vocab_size (lightweight, no weights)
    print("\n=== [base config] ===")
    try:
        cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
        print(f"[config] base_model = {args.base_model}")
        print(f"[config] vocab_size = {getattr(cfg, 'vocab_size', None)}")
        print(f"[config] model_type = {getattr(cfg, 'model_type', None)}")
    except Exception as e:
        print(f"[config] failed to load base config: {e}")

    # Build test texts
    texts = [
        "你好 world",
        "S1->S2",
        "S4->S5",
        "12 34 56",
        "1+1=2",
        "A-B>C",
        "问题：1+1=2\n答案：2",
        "  leading spaces",
        "tabs\tand\nnewlines",
        "0 1 2 3 4 5 6 7 8 9",
    ]

    if args.dataset_jsonl:
        extra = maybe_load_texts_from_jsonl(Path(args.dataset_jsonl), key=args.text_key, limit=args.sample_texts)
        if extra:
            print(f"\n[sample] loaded {len(extra)} texts from {args.dataset_jsonl} key={args.text_key}")
            texts += extra
        else:
            print(f"\n[sample] WARNING: no texts loaded from {args.dataset_jsonl} (key={args.text_key})")

    # Encoding compare
    compare_encodings(t_base, t_ckpt, texts=texts, name1="BASE", name2="CKPT")

    # Strong semantic check: compare vocab dict
    if not args.skip_vocab_dict_compare:
        compare_vocab_mappings(t_base, t_ckpt, name1="BASE", name2="CKPT", max_examples=20)
    else:
        print("\n[vocab mapping compare] skipped (use --skip_vocab_dict_compare=false to enable)")

    # Spot check id->token string
    sample_id_token_checks(t_base, t_ckpt, name1="BASE", name2="CKPT", seed=args.seed, n_random=40)

    print("\n=== DONE ===")
    print("If vocab_dict_equal? is False OR many 'same_ids? False', tokenizer semantic mismatch is very likely.")
    print("If vocab_dict_equal? is True and most 'same_ids? True', tokenizer is likely compatible and we should check reward/eval/exploration next.")

if __name__ == "__main__":
    main()