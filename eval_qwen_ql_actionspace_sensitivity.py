#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Qwen Q-learning checkpoints under:
  1) restricted decoding   = digits + single-space + EOS
  2) unrestricted decoding = full vocab except PAD

Purpose:
  Answer reviewer concern about sensitivity to restricted output/action space.

This script is eval-only. It does NOT train anything.

It reports for each checkpoint x mode:
  - exact success (using raw/applied path mode like your training script)
  - parseable rate
  - accuracy_given_parseable
  - per-bucket metrics
  - per-gap metrics
  - gap>=k aggregates
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


Node = int
Pair = Tuple[int, int]
BucketName = str


# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_ckpt_step(name: str) -> Optional[int]:
    m = re.fullmatch(r"ckpt_ql_(\d+)", name)
    return int(m.group(1)) if m else None


def parse_gap_from_bucket(bucket: str) -> Optional[int]:
    m = re.fullmatch(r"S(\d+)->S(\d+)", bucket)
    if m is None:
        return None
    i, j = map(int, m.groups())
    return j - i


def sanitize_key(s: str) -> str:
    s = s.replace(">=", "ge_")
    s = s.replace("->", "_to_")
    s = s.replace("-", "_")
    s = s.replace(" ", "")
    return s


def prepare_run_dir(base_dir: str) -> Path:
    out_dir = Path(base_dir) / f"eval_{now_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_max_new_tokens(block_size: int, prompt_len: int, desired: int) -> int:
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(f"block_size={block_size} too small for prompt_len={prompt_len}")
    return max(1, min(desired, available))


def top_k_filtering_1d(logits_1d: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits_1d.size(-1):
        return logits_1d
    vals, idx = torch.topk(logits_1d, top_k)
    out = torch.full_like(logits_1d, float("-inf"))
    out.scatter_(0, idx, vals)
    return out


# -------------------------
# data helpers
# -------------------------
def load_pairs(test_file: Path) -> List[Pair]:
    pairs: List[Pair] = []
    for line in test_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def build_node_to_stage(stages: Sequence[Sequence[int]]) -> Dict[int, int]:
    node_to_stage: Dict[int, int] = {}
    for si, nodes in enumerate(stages, start=1):
        for n in nodes:
            n = int(n)
            if n in node_to_stage:
                raise ValueError(f"Node {n} appears in multiple stages.")
            node_to_stage[n] = si
    return node_to_stage


def bucket_for_pair_k(source: int, target: int, node_to_stage: Dict[int, int], K: int) -> Optional[BucketName]:
    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    if si is None or sj is None:
        return None
    if not (1 <= si <= K and 1 <= sj <= K and si < sj):
        return None
    return f"S{si}->S{sj}"


def required_intermediate_stages(si: int, sj: int) -> List[int]:
    if sj <= si + 1:
        return []
    return list(range(si + 1, sj))


# -------------------------
# graph helpers
# -------------------------
def build_int_adjacency(G: nx.DiGraph) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = defaultdict(set)
    for u, v in G.edges():
        try:
            ui = int(str(u))
            vi = int(str(v))
        except Exception:
            continue
        adj[ui].add(vi)
    return dict(adj)


def is_valid_path_k(
    path_nodes: List[int],
    source: int,
    target: int,
    node_to_stage: Dict[int, int],
    adj: Dict[int, Set[int]],
) -> bool:
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if v not in adj.get(int(u), set()):
            return False

    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    if si is None or sj is None or not (si < sj):
        return False

    req = required_intermediate_stages(int(si), int(sj))
    if not req:
        return True

    present = set()
    for n in path_nodes[1:-1]:
        st = node_to_stage.get(int(n))
        if st is not None:
            present.add(int(st))
    return all(r in present for r in req)


# -------------------------
# token action space helpers
# -------------------------
def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_allowed_piece_ascii_digit_or_single_space(piece: str) -> bool:
    if piece is None:
        return False
    if piece == " ":
        return True
    return (len(piece) == 1) and _is_ascii_digit(piece)


def build_allowed_token_mask(tokenizer, device: torch.device) -> torch.Tensor:
    """
    restricted mode:
      only tokens decoding to:
        - one ASCII digit '0'..'9'
        - single space ' '
        - EOS
    """
    V = len(tokenizer)
    allowed = torch.zeros(V, dtype=torch.bool)

    for tid in range(V):
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if _is_allowed_piece_ascii_digit_or_single_space(s):
            allowed[tid] = True

    if tokenizer.eos_token_id is not None:
        allowed[int(tokenizer.eos_token_id)] = True

    return allowed.to(device)


@dataclass
class ActionSpace:
    eos_id: int
    eos_ids: torch.Tensor
    space_ids: torch.Tensor
    digit_ids: Dict[str, torch.Tensor]
    digits_and_eos_ids: torch.Tensor
    digits_space_eos_ids: torch.Tensor
    device: torch.device

    @property
    def all_static_ids(self) -> torch.Tensor:
        return self.digits_space_eos_ids


def build_action_space_single_char(tokenizer, device: torch.device, allowed_token_mask: torch.Tensor) -> ActionSpace:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    allowed_ids = torch.nonzero(allowed_token_mask, as_tuple=False).squeeze(-1).tolist()

    digits: Dict[str, List[int]] = {str(d): [] for d in range(10)}
    spaces: List[int] = []
    has_eos = False

    for tid in allowed_ids:
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s == " ":
            spaces.append(int(tid))
        elif len(s) == 1 and _is_ascii_digit(s):
            digits[s].append(int(tid))
        elif int(tid) == int(eos_id):
            has_eos = True

    if not has_eos:
        raise ValueError("EOS is not included in allowed_token_mask.")
    if len(spaces) == 0:
        raise ValueError("No single-space token id found in tokenizer.")
    for d in range(10):
        if len(digits[str(d)]) == 0:
            raise ValueError(f"No token id found for digit '{d}'.")

    eos_ids = torch.tensor([int(eos_id)], dtype=torch.long, device=device)
    space_ids = torch.tensor(spaces, dtype=torch.long, device=device)
    digit_ids = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in digits.items()}

    parts_no_space = [eos_ids] + [digit_ids[str(d)] for d in range(10)]
    digits_and_eos_ids = torch.cat(parts_no_space, dim=0)
    digits_space_eos_ids = torch.cat(parts_no_space + [space_ids], dim=0)

    return ActionSpace(
        eos_id=int(eos_id),
        eos_ids=eos_ids,
        space_ids=space_ids,
        digit_ids=digit_ids,
        digits_and_eos_ids=digits_and_eos_ids,
        digits_space_eos_ids=digits_space_eos_ids,
        device=device,
    )


def build_full_candidate_ids(tokenizer, device: torch.device) -> torch.Tensor:
    ids = torch.arange(len(tokenizer), dtype=torch.long, device=device)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is not None and (eos_id is None or int(pad_id) != int(eos_id)):
        ids = ids[ids != int(pad_id)]
    return ids


def restricted_candidate_ids(action_space: ActionSpace, pending: str, mask_space_when_pending_empty: bool) -> torch.Tensor:
    if mask_space_when_pending_empty and pending == "":
        return action_space.digits_and_eos_ids
    return action_space.digits_space_eos_ids


def select_next_token_from_candidates(
    logits_1d: torch.Tensor,
    candidate_ids: torch.Tensor,
    temperature: float,
    top_k: int,
) -> int:
    cand_logits = logits_1d.detach().index_select(0, candidate_ids).float()

    if torch.isneginf(cand_logits).all():
        ridx = torch.randint(0, candidate_ids.numel(), (1,), device=candidate_ids.device)
        return int(candidate_ids[ridx].item())

    if top_k > 0:
        K = min(int(top_k), candidate_ids.numel())
        cand_logits = top_k_filtering_1d(cand_logits, K)

    if temperature <= 1e-6:
        ridx = torch.argmax(cand_logits)
        return int(candidate_ids[ridx].item())

    probs = F.softmax(cand_logits / max(float(temperature), 1e-6), dim=-1)
    ridx = torch.multinomial(probs, num_samples=1)
    return int(candidate_ids[ridx].item())


# -------------------------
# parser
# -------------------------
@dataclass
class ParseResult:
    completed_nodes: List[int]
    invalid_char: bool


class NodeStreamParser:
    def __init__(self, node_max: int, node_min: int = 0) -> None:
        self.pending_digits: List[str] = []
        self.node_max = int(node_max)
        self.node_min = int(node_min)
        self.max_digits = max(1, len(str(max(0, self.node_max))))

    def _flush_pending(self) -> Optional[int]:
        if not self.pending_digits:
            return None
        s = "".join(self.pending_digits)
        self.pending_digits.clear()
        try:
            return int(s)
        except Exception:
            return None

    def consume_text(self, piece: str) -> ParseResult:
        completed: List[int] = []
        invalid = False

        for ch in piece:
            if _is_ascii_digit(ch):
                self.pending_digits.append(ch)

                if len(self.pending_digits) > self.max_digits:
                    invalid = True
                    break

                if len(self.pending_digits) == self.max_digits:
                    try:
                        v = int("".join(self.pending_digits))
                        if v > self.node_max:
                            invalid = True
                            break
                    except Exception:
                        invalid = True
                        break

            elif ch == " ":
                node = self._flush_pending()
                if node is not None:
                    completed.append(node)
            else:
                invalid = True
                break

        return ParseResult(completed_nodes=completed, invalid_char=invalid)

    def finalize(self) -> List[int]:
        node = self._flush_pending()
        return [node] if node is not None else []

    def pending_as_str(self) -> str:
        return "".join(self.pending_digits)


# -------------------------
# rollout
# -------------------------
def build_prompt_text(source: int, target: int) -> str:
    return f"{source} {target} {source}"


@torch.no_grad()
def sample_trajectory_hf_eval(
    model,
    tokenizer,
    source: int,
    target: int,
    adj: Dict[int, Set[int]],
    node_to_stage: Dict[int, int],
    K: int,
    device: torch.device,
    block_size: int,
    max_rollout_steps: int,
    temperature: float,
    rollout_top_k: int,
    action_mode: str,  # "restricted" or "unrestricted"
    action_space: Optional[ActionSpace],
    full_candidate_ids: torch.Tensor,
    mask_space_when_pending_empty: bool,
    allow_invalid_continue: bool,
    max_invalid_transitions: int,
    success_path_mode: str,  # "raw" or "applied"
    node_min: int,
    node_max: int,
    reward_hit_target_requires_coverage: bool,
    terminate_on_overshoot: bool,
) -> Dict[str, object]:
    model_was_training = model.training
    model.eval()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    si = node_to_stage.get(int(source))
    sj = node_to_stage.get(int(target))
    target_stage = node_to_stage.get(int(target))
    pair_bucket = bucket_for_pair_k(source, target, node_to_stage=node_to_stage, K=K)

    required_stages = required_intermediate_stages(int(si), int(sj)) if (si is not None and sj is not None and si < sj) else []
    visited_required_stages: Set[int] = set()

    prompt_text = build_prompt_text(source, target) + " "
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)
    max_new = safe_max_new_tokens(block_size, prompt_len, max_rollout_steps)

    traj_ids: List[int] = list(prompt_ids)
    sampled_ids: List[int] = []

    parser = NodeStreamParser(node_max=node_max, node_min=node_min)
    generated_nodes: List[int] = []
    applied_path_nodes: List[int] = [int(source)]
    action_token_texts: List[str] = []

    current_node = int(source)
    invalid_transition = False
    invalid_token = False
    invalid_transition_steps = 0
    hit_target = False

    def covered_all_required() -> bool:
        if not required_stages:
            return True
        return all(s in visited_required_stages for s in required_stages)

    def apply_node_transition(next_node: int) -> bool:
        """
        Returns done flag.
        Mirrors the relevant stopping behavior of your training/eval code.
        """
        nonlocal current_node, invalid_transition, invalid_token, invalid_transition_steps, hit_target

        if next_node < int(node_min) or next_node > int(node_max):
            invalid_token = True
            return True

        next_stage = node_to_stage.get(int(next_node))
        if next_stage is None:
            invalid_token = True
            return True

        if int(next_node) in adj.get(int(current_node), set()):
            if required_stages and int(next_stage) in required_stages:
                visited_required_stages.add(int(next_stage))

            current_node = int(next_node)
            applied_path_nodes.append(int(next_node))

            if terminate_on_overshoot and target_stage is not None and int(next_stage) > int(target_stage):
                return True

            if int(next_node) == int(target):
                if reward_hit_target_requires_coverage and (not covered_all_required()):
                    hit_target = False
                    return True
                hit_target = True
                return True

            return False

        invalid_transition = True
        invalid_transition_steps += 1
        if (not allow_invalid_continue) or (invalid_transition_steps >= max(1, int(max_invalid_transitions))):
            return True
        return False

    with torch.inference_mode():
        input_ids0 = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        out0 = model(input_ids=input_ids0, use_cache=True)
        past = out0.past_key_values
        last_logits = out0.logits[0, -1, :]

    for _step in range(max_new):
        if len(traj_ids) >= block_size - 1:
            break

        if action_mode == "restricted":
            if action_space is None:
                raise ValueError("restricted mode requires action_space.")
            cand_ids = restricted_candidate_ids(
                action_space=action_space,
                pending=parser.pending_as_str(),
                mask_space_when_pending_empty=mask_space_when_pending_empty,
            )
        elif action_mode == "unrestricted":
            cand_ids = full_candidate_ids
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        next_id = select_next_token_from_candidates(
            logits_1d=last_logits,
            candidate_ids=cand_ids,
            temperature=temperature,
            top_k=rollout_top_k,
        )

        traj_ids.append(int(next_id))
        sampled_ids.append(int(next_id))

        piece = tokenizer.decode([int(next_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        action_token_texts.append(piece)

        done = False

        if int(next_id) == int(eos_id):
            for node in parser.finalize():
                generated_nodes.append(int(node))
                done = apply_node_transition(int(node))
                if done:
                    break
            done = True
        else:
            pr = parser.consume_text(piece)
            if pr.invalid_char:
                invalid_token = True
                done = True
            else:
                for node in pr.completed_nodes:
                    generated_nodes.append(int(node))
                    done = apply_node_transition(int(node))
                    if done:
                        break

        if done:
            break

        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([[int(next_id)]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            last_logits = out.logits[0, -1, :]

    # match your original script: ensure an EOS exists at the end
    if (not sampled_ids) or (int(sampled_ids[-1]) != int(eos_id)):
        sampled_ids.append(int(eos_id))
        traj_ids.append(int(eos_id))
        piece = tokenizer.decode([int(eos_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        action_token_texts.append(piece)
        for node in parser.finalize():
            generated_nodes.append(int(node))
            _ = apply_node_transition(int(node))

    raw_path_nodes = [int(source)] + list(map(int, generated_nodes))

    success_applied = is_valid_path_k(
        path_nodes=applied_path_nodes,
        source=int(source),
        target=int(target),
        node_to_stage=node_to_stage,
        adj=adj,
    )
    success_raw = is_valid_path_k(
        path_nodes=raw_path_nodes,
        source=int(source),
        target=int(target),
        node_to_stage=node_to_stage,
        adj=adj,
    )

    if success_path_mode == "raw":
        success_used = success_raw
        path_for_success = raw_path_nodes
    else:
        success_used = success_applied
        path_for_success = applied_path_nodes

    # parseable = syntactically parseable into at least one node, with no invalid chars/tokens
    parseable = (not bool(invalid_token)) and (len(generated_nodes) > 0)

    raw_completion_text = tokenizer.decode(sampled_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if model_was_training:
        model.train()

    return {
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "action_token_texts": action_token_texts,
        "raw_completion_text": raw_completion_text,
        "generated_nodes": generated_nodes,
        "raw_path_nodes": raw_path_nodes,
        "applied_path_nodes": applied_path_nodes,
        "path_nodes": path_for_success,
        "success": bool(success_used),
        "success_raw": bool(success_raw),
        "success_applied": bool(success_applied),
        "parseable": bool(parseable),
        "hit_target": bool(hit_target),
        "invalid_transition": bool(invalid_transition),
        "invalid_token": bool(invalid_token),
        "invalid_transition_steps": int(invalid_transition_steps),
        "bucket": pair_bucket,
    }


# -------------------------
# metrics aggregation
# -------------------------
def new_counter() -> Dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "correct_raw": 0,
        "correct_applied": 0,
        "parseable": 0,
        "invalid_token": 0,
        "invalid_transition": 0,
    }


def finalize_counter(c: Dict[str, int]) -> Dict[str, float]:
    total = int(c["total"])
    correct = int(c["correct"])
    correct_raw = int(c["correct_raw"])
    correct_applied = int(c["correct_applied"])
    parseable = int(c["parseable"])
    invalid_token = int(c["invalid_token"])
    invalid_transition = int(c["invalid_transition"])

    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total if total else 0.0),
        "correct_raw": correct_raw,
        "accuracy_raw": (correct_raw / total if total else 0.0),
        "correct_applied": correct_applied,
        "accuracy_applied": (correct_applied / total if total else 0.0),
        "parseable": parseable,
        "parseable_rate": (parseable / total if total else 0.0),
        "accuracy_given_parseable": (correct / parseable if parseable else 0.0),
        "invalid_token_count": invalid_token,
        "invalid_token_rate": (invalid_token / total if total else 0.0),
        "invalid_transition_count": invalid_transition,
        "invalid_transition_rate": (invalid_transition / total if total else 0.0),
    }


@torch.no_grad()
def evaluate_checkpoint(
    model,
    tokenizer,
    pairs: List[Pair],
    node_to_stage: Dict[int, int],
    K: int,
    adj: Dict[int, Set[int]],
    device: torch.device,
    block_size: int,
    max_rollout_steps: int,
    max_eval_pairs: int,
    eval_temperature: float,
    rollout_top_k: int,
    action_mode: str,
    action_space: Optional[ActionSpace],
    full_candidate_ids: torch.Tensor,
    mask_space_when_pending_empty: bool,
    allow_invalid_continue: bool,
    max_invalid_transitions: int,
    success_path_mode: str,
    node_min: int,
    node_max: int,
    reward_hit_target_requires_coverage: bool,
    terminate_on_overshoot: bool,
) -> Dict[str, object]:
    model.eval()

    selected_pairs = pairs if max_eval_pairs <= 0 else pairs[:max_eval_pairs]
    bucket_names = [f"S{i}->S{j}" for i in range(1, K + 1) for j in range(i + 1, K + 1)]

    counts_by_bucket: Dict[str, Dict[str, int]] = {bn: new_counter() for bn in bucket_names}

    t0 = time.perf_counter()
    for idx, (s, t) in enumerate(selected_pairs, start=1):
        b = bucket_for_pair_k(s, t, node_to_stage=node_to_stage, K=K)
        if b is None:
            continue

        traj = sample_trajectory_hf_eval(
            model=model,
            tokenizer=tokenizer,
            source=s,
            target=t,
            adj=adj,
            node_to_stage=node_to_stage,
            K=K,
            device=device,
            block_size=block_size,
            max_rollout_steps=max_rollout_steps,
            temperature=eval_temperature,
            rollout_top_k=rollout_top_k,
            action_mode=action_mode,
            action_space=action_space,
            full_candidate_ids=full_candidate_ids,
            mask_space_when_pending_empty=mask_space_when_pending_empty,
            allow_invalid_continue=allow_invalid_continue,
            max_invalid_transitions=max_invalid_transitions,
            success_path_mode=success_path_mode,
            node_min=node_min,
            node_max=node_max,
            reward_hit_target_requires_coverage=reward_hit_target_requires_coverage,
            terminate_on_overshoot=terminate_on_overshoot,
        )

        c = counts_by_bucket[b]
        c["total"] += 1
        c["correct"] += int(bool(traj["success"]))
        c["correct_raw"] += int(bool(traj["success_raw"]))
        c["correct_applied"] += int(bool(traj["success_applied"]))
        c["parseable"] += int(bool(traj["parseable"]))
        c["invalid_token"] += int(bool(traj["invalid_token"]))
        c["invalid_transition"] += int(bool(traj["invalid_transition"]))

        if idx % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{action_mode}] eval progress: {idx}/{len(selected_pairs)} pairs, elapsed={elapsed/60.0:.2f} min")

    by_bucket = {bn: finalize_counter(counts_by_bucket[bn]) for bn in bucket_names}

    # aggregate by exact gap
    gap_counters: Dict[int, Dict[str, int]] = defaultdict(new_counter)
    overall_counter = new_counter()

    for bn in bucket_names:
        g = parse_gap_from_bucket(bn)
        c = counts_by_bucket[bn]
        if g is not None:
            for k in overall_counter.keys():
                gap_counters[g][k] += int(c[k])
        for k in overall_counter.keys():
            overall_counter[k] += int(c[k])

    by_gap: Dict[str, Dict[str, float]] = {}
    for g in sorted(gap_counters.keys()):
        by_gap[f"gap{g}"] = finalize_counter(gap_counters[g])

    # aggregate gap>=k
    gap_ge: Dict[str, Dict[str, float]] = {}
    max_gap = max((parse_gap_from_bucket(bn) or 0) for bn in bucket_names)
    for thr in range(2, max_gap + 1):
        agg = new_counter()
        for g, c in gap_counters.items():
            if g >= thr:
                for k in agg.keys():
                    agg[k] += int(c[k])
        gap_ge[f"gap>={thr}"] = finalize_counter(agg)

    overall = finalize_counter(overall_counter)

    return {
        "mode": action_mode,
        "num_eval_pairs": len(selected_pairs),
        "by_bucket": by_bucket,
        "by_gap": by_gap,
        "gap_ge": gap_ge,
        "overall": overall,
    }


# -------------------------
# model/tokenizer loading
# -------------------------
def maybe_resize_to_tokenizer_vocab(model, tokenizer_len: int) -> None:
    cur = int(model.get_input_embeddings().weight.size(0))
    if cur == int(tokenizer_len):
        return
    print(f"[vocab] resizing embeddings {cur} -> {int(tokenizer_len)}")
    model.resize_token_embeddings(int(tokenizer_len))
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass


def load_model_for_eval(
    base_model: str,
    ckpt_dir: Path,
    tokenizer_len: int,
    device: torch.device,
    bf16: bool,
    fp16: bool,
    trust_remote_code: bool,
):
    if bf16 and device.type == "cuda":
        torch_dtype = torch.bfloat16
    elif fp16 and device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if (ckpt_dir / "adapter_config.json").exists():
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but checkpoint is a LoRA adapter.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(trust_remote_code),
            low_cpu_mem_usage=True,
        )
        maybe_resize_to_tokenizer_vocab(base, tokenizer_len)
        model = PeftModel.from_pretrained(base, str(ckpt_dir), is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch_dtype,
            trust_remote_code=bool(trust_remote_code),
            low_cpu_mem_usage=True,
        )
        maybe_resize_to_tokenizer_vocab(model, tokenizer_len)

    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(tokenizer_source: Path, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        use_fast=True,
        trust_remote_code=bool(trust_remote_code),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer missing eos_token_id.")
    if tokenizer.pad_token_id is None:
        # fallback only if necessary
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -------------------------
# checkpoint discovery
# -------------------------
def resolve_ckpt_dirs(ckpt_root: Path, ckpt_steps: str) -> List[Path]:
    all_dirs = []
    for p in ckpt_root.iterdir():
        if p.is_dir():
            step = parse_ckpt_step(p.name)
            if step is not None:
                all_dirs.append((step, p.resolve()))
    if not all_dirs:
        raise FileNotFoundError(f"No ckpt_ql_* folders found under {ckpt_root}")

    all_dirs = sorted(all_dirs, key=lambda x: x[0])

    spec = ckpt_steps.strip().lower()
    if spec == "all":
        return [p for _, p in all_dirs]
    if spec == "final":
        return [all_dirs[-1][1]]

    want_steps = [int(x.strip()) for x in ckpt_steps.split(",") if x.strip()]
    step_to_dir = {step: p for step, p in all_dirs}

    out = []
    for step in want_steps:
        if step not in step_to_dir:
            raise FileNotFoundError(f"Requested ckpt_ql_{step} not found under {ckpt_root}")
        out.append(step_to_dir[step])
    return out


# -------------------------
# summary flattening
# -------------------------
def add_metric_fields(row: Dict[str, object], prefix: str, metric: Dict[str, float]) -> None:
    p = sanitize_key(prefix)
    for k, v in metric.items():
        row[f"{p}_{k}"] = v


def flatten_record(ckpt_step: int, ckpt_dir: str, mode: str, results: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "ckpt_step": ckpt_step,
        "ckpt_dir": ckpt_dir,
        "mode": mode,
        "num_eval_pairs": results["num_eval_pairs"],
    }

    add_metric_fields(row, "overall", results["overall"])

    for k, v in results["by_gap"].items():
        add_metric_fields(row, k, v)

    for k, v in results["gap_ge"].items():
        add_metric_fields(row, k, v)

    for k, v in results["by_bucket"].items():
        add_metric_fields(row, k, v)

    return row


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_brief(results: Dict[str, object], ckpt_step: int, mode: str) -> None:
    ov = results["overall"]
    ge2 = results["gap_ge"].get("gap>=2", None)
    msg = (
        f"[ckpt={ckpt_step:>4}][mode={mode:<12}] "
        f"overall_acc={ov['accuracy']:.4f}  "
        f"parseable={ov['parseable_rate']:.4f}  "
        f"acc|parseable={ov['accuracy_given_parseable']:.4f}"
    )
    if ge2 is not None:
        msg += (
            f"  gap>=2_acc={ge2['accuracy']:.4f}  "
            f"gap>=2_parseable={ge2['parseable_rate']:.4f}  "
            f"gap>=2_acc|parseable={ge2['accuracy_given_parseable']:.4f}"
        )
    print(msg)


# -------------------------
# args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate Qwen QL checkpoints under restricted vs unrestricted decoding")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt_root", type=str, default=None, help="Root folder containing ckpt_ql_* subfolders")
    group.add_argument("--ckpt_dir", type=str, default=None, help="Single ckpt_ql_* folder")

    p.add_argument("--ckpt_steps", type=str, default="final",
                   help='For --ckpt_root only. Examples: "final", "all", "900,1000,1100,1200"')

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--max_rollout_steps", type=int, default=64)
    p.add_argument("--max_eval_pairs", type=int, default=1200)
    p.add_argument("--eval_temperature", type=float, default=1e-3)
    p.add_argument("--rollout_top_k", type=int, default=0)

    p.add_argument("--action_modes", type=str, default="restricted,unrestricted",
                   help='Comma-separated subset of {"restricted","unrestricted"}')

    p.add_argument("--mask_space_when_pending_empty", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_invalid_continue", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_invalid_transitions", type=int, default=3)
    p.add_argument("--success_path_mode", choices=["raw", "applied"], default="raw")

    p.add_argument("--node_min", type=int, default=0)
    p.add_argument("--node_max", type=int, default=149)

    p.add_argument("--reward_hit_target_requires_coverage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--terminate_on_overshoot", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--out_dir", type=str, default="eval_qwen_ql_actionspace")

    return p.parse_args()


# -------------------------
# main
# -------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    data_dir = Path(args.data_dir).resolve()
    test_txt = data_dir / "test.txt"
    stage_info_pkl = data_dir / "stage_info.pkl"
    graph_p = data_dir / "composition_graph.graphml"
    meta_p = data_dir / "meta.pkl"

    if not test_txt.exists():
        raise FileNotFoundError(test_txt)
    if not stage_info_pkl.exists():
        raise FileNotFoundError(stage_info_pkl)
    if not graph_p.exists():
        raise FileNotFoundError(graph_p)

    if args.ckpt_dir is not None:
        ckpt_dirs = [Path(args.ckpt_dir).resolve()]
    else:
        ckpt_dirs = resolve_ckpt_dirs(Path(args.ckpt_root).resolve(), args.ckpt_steps)

    modes = [x.strip() for x in args.action_modes.split(",") if x.strip()]
    valid_modes = {"restricted", "unrestricted"}
    if any(m not in valid_modes for m in modes):
        raise ValueError(f"--action_modes must be subset of {valid_modes}")
    modes = list(dict.fromkeys(modes))  # dedupe, keep order

    # load data/meta
    if meta_p.exists():
        meta = pickle.load(open(meta_p, "rb"))
    else:
        # fallback to first ckpt's data_meta.pkl
        fallback = ckpt_dirs[0] / "data_meta.pkl"
        if not fallback.exists():
            raise FileNotFoundError(f"Neither {meta_p} nor {fallback} exists.")
        meta = pickle.load(open(fallback, "rb"))
    block_size = int(meta.get("block_size", 63))

    stage_info = pickle.load(open(stage_info_pkl, "rb"))
    stages: List[List[int]] = stage_info["stages"]
    K = len(stages)
    node_to_stage = build_node_to_stage(stages)

    G = nx.read_graphml(graph_p)
    adj = build_int_adjacency(G)
    eval_pairs = load_pairs(test_txt)

    run_dir = prepare_run_dir(args.out_dir)
    jsonl_path = run_dir / "results.jsonl"
    csv_path = run_dir / "summary.csv"
    config_path = run_dir / "eval_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[run_dir] {run_dir}")
    print(f"[data] data_dir={data_dir}")
    print(f"[data] K={K}, #eval_pairs_total={len(eval_pairs)}, block_size={block_size}")
    print(f"[ckpts] {len(ckpt_dirs)} checkpoints selected:")
    for p in ckpt_dirs:
        print(f"  - {p}")

    # tokenizer: load once from first checkpoint if tokenizer exists there, else from data_dir/tokenizer
    tok_source = ckpt_dirs[0] if (ckpt_dirs[0] / "tokenizer.json").exists() else (data_dir / "tokenizer")
    if not tok_source.exists():
        raise FileNotFoundError(f"Tokenizer source not found: {tok_source}")
    tokenizer = load_tokenizer(tok_source, trust_remote_code=args.trust_remote_code)
    print(f"[tokenizer] loaded from {tok_source}")
    print(f"[tokenizer] vocab={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}")

    # action spaces: build once
    allowed_token_mask = build_allowed_token_mask(tokenizer, device=device)
    kept = int(allowed_token_mask.sum().item())
    print(f"[restricted-mask] allowed={kept}/{len(tokenizer)} tokens")
    action_space = build_action_space_single_char(tokenizer, device=device, allowed_token_mask=allowed_token_mask)
    full_candidate_ids = build_full_candidate_ids(tokenizer, device=device)
    print(f"[unrestricted-mask] allowed={int(full_candidate_ids.numel())}/{len(tokenizer)} tokens (PAD removed if distinct)")

    summary_rows: List[Dict[str, object]] = []

    t0_all = time.perf_counter()
    for ckpt_dir in ckpt_dirs:
        step = parse_ckpt_step(ckpt_dir.name)
        if step is None:
            raise ValueError(f"Unexpected checkpoint dir name: {ckpt_dir.name}")

        print("\n" + "=" * 100)
        print(f"[load] checkpoint={ckpt_dir} (step={step})")

        model = load_model_for_eval(
            base_model=args.base_model,
            ckpt_dir=ckpt_dir,
            tokenizer_len=len(tokenizer),
            device=device,
            bf16=args.bf16,
            fp16=args.fp16,
            trust_remote_code=args.trust_remote_code,
        )

        try:
            for mode in modes:
                print(f"[eval] step={step}, mode={mode}")
                t0 = time.perf_counter()

                results = evaluate_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    pairs=eval_pairs,
                    node_to_stage=node_to_stage,
                    K=K,
                    adj=adj,
                    device=device,
                    block_size=block_size,
                    max_rollout_steps=args.max_rollout_steps,
                    max_eval_pairs=args.max_eval_pairs,
                    eval_temperature=args.eval_temperature,
                    rollout_top_k=args.rollout_top_k,
                    action_mode=mode,
                    action_space=action_space if mode == "restricted" else None,
                    full_candidate_ids=full_candidate_ids,
                    mask_space_when_pending_empty=args.mask_space_when_pending_empty,
                    allow_invalid_continue=args.allow_invalid_continue,
                    max_invalid_transitions=args.max_invalid_transitions,
                    success_path_mode=args.success_path_mode,
                    node_min=args.node_min,
                    node_max=args.node_max,
                    reward_hit_target_requires_coverage=args.reward_hit_target_requires_coverage,
                    terminate_on_overshoot=args.terminate_on_overshoot,
                )

                elapsed = time.perf_counter() - t0
                print_brief(results, ckpt_step=step, mode=mode)
                print(f"[done] step={step}, mode={mode}, elapsed={elapsed/60.0:.2f} min")

                record = {
                    "ckpt_step": step,
                    "ckpt_dir": str(ckpt_dir),
                    "mode": mode,
                    "results": results,
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                summary_rows.append(flatten_record(
                    ckpt_step=step,
                    ckpt_dir=str(ckpt_dir),
                    mode=mode,
                    results=results,
                ))

                write_csv(summary_rows, csv_path)

        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t0_all
    print("\n" + "=" * 100)
    print(f"[finished] total elapsed = {total_elapsed/60.0:.2f} min")
    print(f"[saved] jsonl = {jsonl_path}")
    print(f"[saved] csv   = {csv_path}")
    print(f"[saved] config= {config_path}")


if __name__ == "__main__":
    main()