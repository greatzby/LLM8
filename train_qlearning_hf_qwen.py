#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q-learning fine-tuning for graph composition datasets (HF/Qwen tokenizer).

Key updates in this final version:
1) Rollout acceleration with KV cache (use_cache=True) => huge speedup, same semantics in eval mode.
2) Epsilon exploration decoupled from rollout_top_k:
   - rollout_top_k can stay 1 (stable greedy)
   - epsilon exploration samples uniformly from top-N logits (default 20) => exploration without "top_k=20 divergence"
3) Tokenizer loading: prefer tokenizer from --sft_dir if present, else fallback to data_dir/tokenizer.

Patch (2025-12-23):
- Fix PEFT adapter load "size mismatch embed_tokens/lm_head" by resizing the BASE model's
  token embeddings to match the tokenizer vocab size *before* calling PeftModel.from_pretrained().
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel, get_peft_model, LoraConfig, TaskType
except Exception:
    PeftModel = None
    get_peft_model = None
    LoraConfig = None
    TaskType = None


Node = int
Pair = Tuple[int, int]
BucketName = str


# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_output_dir(base_dir: str) -> Path:
    out_dir = Path(base_dir) / f"ql_{now_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def current_temperature(iteration: int, args: argparse.Namespace) -> float:
    if args.temp_warmup_iters <= 0 or args.rollout_temp_start == args.rollout_temp_end:
        return args.rollout_temp_end
    ratio = min(1.0, iteration / max(1, args.temp_warmup_iters))
    return args.rollout_temp_start + ratio * (args.rollout_temp_end - args.rollout_temp_start)


def current_epsilon(iteration: int, args: argparse.Namespace) -> float:
    if args.epsilon_warmup_iters <= 0 or args.epsilon_start == args.epsilon_end:
        return args.epsilon_end
    ratio = min(1.0, iteration / max(1, args.epsilon_warmup_iters))
    return args.epsilon_start + ratio * (args.epsilon_end - args.epsilon_start)


def current_kl_coef(iteration: int, args: argparse.Namespace) -> float:
    if args.kl_coef <= 0.0:
        return 0.0
    if iteration <= args.kl_warmup_iters:
        return args.kl_coef
    if args.kl_anneal_iters <= 0:
        return args.kl_coef
    progress = min(1.0, (iteration - args.kl_warmup_iters) / max(1, args.kl_anneal_iters))
    return max(0.0, args.kl_coef * (1.0 - progress))


def load_pairs_unique(train_file: Path) -> List[Pair]:
    seen = set()
    pairs: List[Pair] = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        s, t = int(parts[0]), int(parts[1])
        key = (s, t)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def bucket_for_pair(source: int, target: int, stages: Sequence[Sequence[int]]) -> Optional[BucketName]:
    if len(stages) < 3:
        return None
    S1, S2, S3 = stages[:3]
    if source in S1 and target in S2:
        return "S1->S2"
    if source in S2 and target in S3:
        return "S2->S3"
    if source in S1 and target in S3:
        return "S1->S3"
    return None


def is_valid_path(
    path_nodes: List[int],
    source: int,
    target: int,
    stages: Sequence[Sequence[int]],
    graph: nx.DiGraph,
) -> bool:
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            return False

    # S1->S3 must pass some S2 node
    if len(stages) >= 3:
        S1, S2, S3 = stages[:3]
        if source in S1 and target in S3:
            if not any(node in S2 for node in path_nodes[1:-1]):
                return False
    return True


def safe_max_new_tokens(block_size: int, prompt_len: int, desired: int) -> int:
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(f"block_size={block_size} too small for prompt_len={prompt_len}")
    return max(1, min(desired, available))


def top_k_filtering(logits: Tensor, top_k: int) -> Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    vals, idx = torch.topk(logits, top_k)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(0, idx, vals)
    return mask


def select_next_token(
    logits_1d: Tensor,
    temperature: float,
    top_k: int,
    epsilon: float,
    epsilon_explore_top_k: int,
) -> int:
    """
    logits_1d: [V]

    Final behavior:
    - With prob epsilon: uniformly pick one token from TOP-N logits (N=epsilon_explore_top_k),
      independent of rollout_top_k. This makes epsilon meaningful even if rollout_top_k=1.
    - Otherwise: (optional) top-k filtering (rollout_top_k), then greedy or temperature sampling.
    """
    logits = logits_1d.detach()

    # epsilon exploration (decoupled from rollout_top_k)
    if epsilon > 0.0 and random.random() < epsilon:
        K = int(epsilon_explore_top_k)
        if K <= 0 or K >= logits.numel():
            return int(torch.randint(0, logits.numel(), (1,), device=logits.device).item())
        _, idx = torch.topk(logits, K)
        ridx = idx[torch.randint(0, idx.numel(), (1,), device=logits.device)]
        return int(ridx.item())

    # exploitation
    if top_k > 0:
        logits = top_k_filtering(logits, top_k)

    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# -------------------------
# streaming node parser
# -------------------------
@dataclass
class ParseResult:
    completed_nodes: List[int]
    invalid_char: bool


class NodeStreamParser:
    """
    Consume decoded text pieces and output integer nodes separated by whitespace.
    Valid chars: digits and whitespace. Any other char => invalid_char=True.
    """

    def __init__(self) -> None:
        self.pending_digits: List[str] = []

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
            if ch.isdigit():
                self.pending_digits.append(ch)
            elif ch.isspace():
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


# -------------------------
# model forward helpers
# -------------------------
def forward_logits(model, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits


def action_logprobs_from_logits(logits: Tensor, actions: Tensor) -> Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


# -------------------------
# rollout
# -------------------------
def build_prompt_text(source: int, target: int) -> str:
    return f"{source} {target} {source}"


def assemble_full_path(source: int, generated_nodes: Sequence[int]) -> List[int]:
    return [source] + list(generated_nodes)


@torch.no_grad()
def sample_trajectory_hf(
    model,
    tokenizer,
    source: int,
    target: int,
    graph: nx.DiGraph,
    stages: Sequence[Sequence[int]],
    stage_sets: Dict[str, set[int]],
    pair_bucket: Optional[str],
    args: argparse.Namespace,
    device: torch.device,
    block_size: int,
    temperature: float,
    epsilon: float,
) -> Dict[str, object]:
    """
    Token-level rollout with KV cache acceleration.
    Reward assigned to the token that completes a node (or eos).
    """
    model_was_training = model.training
    model.eval()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run rollout safely.")

    prompt_text = build_prompt_text(source, target) + " "
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    max_new = safe_max_new_tokens(block_size, prompt_len, args.max_rollout_steps)

    traj_ids: List[int] = list(prompt_ids)
    sampled_ids: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []

    parser = NodeStreamParser()
    generated_nodes: List[int] = []

    current_node = source
    visited_nodes = {source}

    hit_target = False
    invalid_transition = False
    invalid_token = False
    visited_stage2 = False
    stage_bridge_rewarded = False
    valid_transition_steps = 0
    invalid_transition_steps = 0

    allow_continue = bool(args.allow_invalid_continue)
    max_invalid = max(1, int(args.max_invalid_transitions)) if allow_continue else 1

    def apply_node_transition(next_node: int) -> Tuple[float, bool]:
        nonlocal current_node, hit_target, invalid_transition, visited_stage2
        nonlocal stage_bridge_rewarded, valid_transition_steps, invalid_transition_steps

        if next_node < args.node_min or next_node > args.node_max:
            return (-args.reward_invalid_token, True)

        reward_delta = 0.0
        done = False

        adjacency = graph.has_edge(str(current_node), str(next_node))
        if adjacency:
            valid_transition_steps += 1
            reward_delta += args.reward_valid_transition

            if next_node in stage_sets.get("S2", set()):
                visited_stage2 = True

            if pair_bucket == "S1->S3" and args.reward_stage_bridge > 0.0:
                if (not stage_bridge_rewarded) and next_node in stage_sets.get("S2", set()):
                    reward_delta += args.reward_stage_bridge
                    if args.reward_stage_bridge_only_once:
                        stage_bridge_rewarded = True

            if pair_bucket == "S1->S2" and args.penalty_stage2_detour > 0.0:
                if next_node != target and next_node in stage_sets.get("S2", set()):
                    reward_delta -= args.penalty_stage2_detour

            if pair_bucket == "S2->S3" and args.penalty_stage3_detour > 0.0:
                if next_node != target and next_node in stage_sets.get("S3", set()):
                    reward_delta -= args.penalty_stage3_detour

            if args.penalty_repeat_node > 0.0 and next_node in visited_nodes and next_node != target:
                reward_delta -= args.penalty_repeat_node

            visited_nodes.add(next_node)
            current_node = next_node

            if next_node == target:
                reward_delta += args.reward_hit_target
                hit_target = True
                done = True
        else:
            invalid_transition = True
            invalid_transition_steps += 1
            reward_delta -= args.reward_invalid_transition
            if (not allow_continue) or (invalid_transition_steps >= max_invalid):
                done = True

        return reward_delta, done

    # ---- KV cache init: run prompt once
    with torch.inference_mode():
        input_ids0 = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        out0 = model(input_ids=input_ids0, use_cache=True)
        past = out0.past_key_values
        last_logits = out0.logits[0, -1, :]  # [V]

    # rollout token-by-token
    for _step in range(max_new):
        if len(traj_ids) >= block_size - 1:
            break

        next_id = select_next_token(
            logits_1d=last_logits,
            temperature=temperature,
            top_k=args.rollout_top_k,
            epsilon=epsilon,
            epsilon_explore_top_k=args.epsilon_explore_top_k,
        )

        traj_ids.append(next_id)
        sampled_ids.append(next_id)

        reward = (-args.step_penalty) if args.step_penalty != 0.0 else 0.0
        done = False

        if next_id == eos_id:
            for node in parser.finalize():
                generated_nodes.append(node)
                r_delta, d2 = apply_node_transition(node)
                reward += r_delta
                if d2:
                    done = True
                    break
            reward += args.reward_stop
            done = True
        else:
            piece = tokenizer.decode([next_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            pr = parser.consume_text(piece)

            if pr.invalid_char:
                invalid_token = True
                reward -= args.reward_invalid_token
                done = True
            else:
                for node in pr.completed_nodes:
                    generated_nodes.append(node)
                    r_delta, d2 = apply_node_transition(node)
                    reward += r_delta
                    if d2:
                        done = True
                        break

        rewards.append(float(reward))
        dones.append(bool(done))
        if done:
            break

        # advance KV cache by feeding only the last generated token
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([[next_id]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            last_logits = out.logits[0, -1, :]

    # if not ended, append eos manually
    if not sampled_ids or sampled_ids[-1] != eos_id:
        sampled_ids.append(eos_id)
        traj_ids.append(eos_id)

        extra_reward = (-args.step_penalty if args.step_penalty != 0.0 else 0.0)
        for node in parser.finalize():
            generated_nodes.append(node)
            r_delta, _ = apply_node_transition(node)
            extra_reward += r_delta
        extra_reward += args.reward_stop

        rewards.append(float(extra_reward))
        dones.append(True)

    full_path_nodes = assemble_full_path(source, generated_nodes)
    success = is_valid_path(full_path_nodes, source, target, stages, graph)

    if args.reward_type == "outcome":
        adjusted = [0.0 for _ in rewards]
        if success and hit_target:
            adjusted[-1] = float(args.reward_hit_target)
        rewards = adjusted

    if model_was_training:
        model.train()

    return {
        "prompt_ids": prompt_ids,
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "rewards": rewards,
        "dones": dones,
        "generated_nodes": generated_nodes,
        "path_nodes": full_path_nodes,
        "success": bool(success),
        "hit_target": bool(hit_target and success),
        "invalid_transition": bool(invalid_transition),
        "invalid_token": bool(invalid_token),
        "visited_stage2": bool(visited_stage2 or stage_bridge_rewarded),
        "valid_transition_steps": int(valid_transition_steps),
        "invalid_transition_steps": int(invalid_transition_steps),
        "bucket": pair_bucket,
    }


# -------------------------
# loss (token-level Q-learning)
# -------------------------
def q_learning_loss_single_traj(
    model,
    target_model,
    traj_ids: List[int],
    prompt_len: int,
    actions: List[int],
    rewards: List[float],
    dones: List[bool],
    device: torch.device,
    gamma: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if len(actions) == 0:
        raise ValueError("Empty actions.")

    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y_ids = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    logits = forward_logits(model, x_ids)[0]  # [L, V]
    start_idx = prompt_len - 1
    num_steps = len(actions)

    if start_idx + num_steps > logits.size(0):
        raise ValueError("Action segment exceeds logits length. Check block_size/prompt_len.")

    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.bool, device=device)

    expected = y_ids[0, start_idx : start_idx + num_steps]
    if expected.numel() != actions_t.numel() or not torch.equal(expected, actions_t):
        raise ValueError("Teacher-forced expected actions != sampled actions (alignment bug).")

    policy_seg = logits[start_idx : start_idx + num_steps, :]  # [T, V]
    q_selected = policy_seg.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)  # [T]
    action_logp = action_logprobs_from_logits(policy_seg, actions_t)  # [T]

    with torch.no_grad():
        if target_model is None:
            target_logits = logits.detach()
        else:
            target_logits = forward_logits(target_model, x_ids)[0].detach()

        target_seg = target_logits[start_idx : start_idx + num_steps, :]
        next_max = torch.zeros_like(q_selected)

        if num_steps > 1:
            next_max[:-1] = target_seg[1:].max(dim=-1).values
        next_max[-1] = 0.0
        next_max = next_max * (~dones_t)

        targets = rewards_t + gamma * next_max

    loss = F.mse_loss(q_selected.float(), targets.float(), reduction="mean")
    return loss, q_selected.detach(), targets.detach(), action_logp.detach()


def kl_loss_action_level(
    policy_model,
    ref_model,
    traj_ids: List[int],
    prompt_len: int,
    actions: List[int],
    device: torch.device,
) -> Tensor:
    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    logits_pi = forward_logits(policy_model, x_ids)[0]
    logits_ref = forward_logits(ref_model, x_ids)[0]

    start_idx = prompt_len - 1
    T = len(actions)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)

    seg_pi = logits_pi[start_idx : start_idx + T, :]
    seg_ref = logits_ref[start_idx : start_idx + T, :]

    logp_pi = action_logprobs_from_logits(seg_pi, actions_t)
    logp_ref = action_logprobs_from_logits(seg_ref, actions_t)
    return (logp_pi - logp_ref).mean()


def kl_loss_full_vocab(
    policy_model,
    ref_model,
    traj_ids: List[int],
    prompt_len: int,
    actions_len: int,
    device: torch.device,
) -> Tensor:
    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    logits_pi = forward_logits(policy_model, x_ids)[0]
    logits_ref = forward_logits(ref_model, x_ids)[0]

    start_idx = prompt_len - 1
    seg_pi = logits_pi[start_idx : start_idx + actions_len, :]
    seg_ref = logits_ref[start_idx : start_idx + actions_len, :]

    return F.kl_div(
        F.log_softmax(seg_pi, dim=-1),
        F.softmax(seg_ref, dim=-1),
        reduction="batchmean",
    )


# -------------------------
# eval
# -------------------------
@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    pairs: List[Pair],
    stages: Sequence[Sequence[int]],
    graph: nx.DiGraph,
    device: torch.device,
    args: argparse.Namespace,
    block_size: int,
    max_pairs: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()

    stage_sets = {
        "S1": set(stages[0]) if len(stages) > 0 else set(),
        "S2": set(stages[1]) if len(stages) > 1 else set(),
        "S3": set(stages[2]) if len(stages) > 2 else set(),
    }

    buckets: Dict[BucketName, List[Pair]] = {"S1->S2": [], "S2->S3": [], "S1->S3": []}
    if max_pairs <= 0:
        return {
            "S1->S2": {"correct": 0, "total": 0, "accuracy": 0.0},
            "S2->S3": {"correct": 0, "total": 0, "accuracy": 0.0},
            "S1->S3": {"correct": 0, "total": 0, "accuracy": 0.0},
            "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
        }

    for s, t in pairs[:max_pairs]:
        b = bucket_for_pair(s, t, stages)
        if b:
            buckets[b].append((s, t))

    total_correct = 0
    total_cases = 0
    results: Dict[str, Dict[str, float]] = {}

    for bname, bpairs in buckets.items():
        correct = 0
        for s, t in bpairs:
            traj = sample_trajectory_hf(
                model=model,
                tokenizer=tokenizer,
                source=s,
                target=t,
                graph=graph,
                stages=stages,
                stage_sets=stage_sets,
                pair_bucket=bname,
                args=args,
                device=device,
                block_size=block_size,
                temperature=args.eval_temperature,
                epsilon=0.0,
            )
            if traj["success"]:
                correct += 1

        total_correct += correct
        total_cases += len(bpairs)
        acc = correct / len(bpairs) if bpairs else 0.0
        results[bname] = {"correct": correct, "total": len(bpairs), "accuracy": acc}

    overall = total_correct / total_cases if total_cases else 0.0
    results["overall"] = {"correct": total_correct, "total": total_cases, "accuracy": overall}
    model.train()
    return results


# -------------------------
# model loading (supports LoRA adapters)
# -------------------------
def maybe_wrap_lora(model, args: argparse.Namespace):
    if args.lora_r <= 0:
        return model
    if get_peft_model is None:
        raise RuntimeError("peft not installed but lora_r>0 was set.")
    lconf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
    )
    return get_peft_model(model, lconf)


def load_policy_model_and_ref(
    base_model: str,
    sft_dir: str,
    device: torch.device,
    args: argparse.Namespace,
    vocab_size: int,
):
    """
    IMPORTANT:
    If sft_dir is a PEFT adapter dir, we must ensure the *base* model's embedding/lm_head
    shapes match the tokenizer's vocab size before calling PeftModel.from_pretrained().
    Otherwise you'll see:
      size mismatch for embed_tokens.weight / lm_head.weight
    """
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    def _load_base():
        kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if args.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": device.index if device.type == "cuda" else "cpu"}
        return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    def _maybe_resize_to_tokenizer_vocab(m, name: str) -> None:
        if vocab_size is None or vocab_size <= 0:
            return
        try:
            cur = int(m.get_input_embeddings().weight.size(0))
        except Exception:
            return
        if cur == int(vocab_size):
            return

        print(f"[vocab] {name}: resizing token embeddings {cur} -> {int(vocab_size)} (to match tokenizer)")

        # Note: on some 4bit setups, resize may fail; give a clear error.
        try:
            m.resize_token_embeddings(int(vocab_size))
            # ensure tying stays consistent (some models need this explicitly)
            if hasattr(m, "tie_weights"):
                try:
                    m.tie_weights()
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(
                f"Failed to resize_token_embeddings for {name} (cur={cur}, target={vocab_size}). "
                f"If you are using --load_in_4bit, try disabling it for adapter loading. Original error: {e}"
            )

    sft_path = Path(sft_dir)
    has_adapter = (sft_path / "adapter_config.json").exists()

    if has_adapter:
        if PeftModel is None:
            raise RuntimeError("peft not installed but adapter_config.json exists.")
        base = _load_base()
        _maybe_resize_to_tokenizer_vocab(base, name="base(policy)")
        policy = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    else:
        kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if args.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": device.index if device.type == "cuda" else "cpu"}
        policy = AutoModelForCausalLM.from_pretrained(sft_dir, **kwargs)
        _maybe_resize_to_tokenizer_vocab(policy, name="policy(full)")
        policy = maybe_wrap_lora(policy, args)

    ref = None
    if args.kl_coef > 0.0:
        if has_adapter:
            base2 = _load_base()
            _maybe_resize_to_tokenizer_vocab(base2, name="base(ref)")
            ref = PeftModel.from_pretrained(base2, sft_dir, is_trainable=False)
        else:
            kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=bool(args.trust_remote_code),
            )
            if args.load_in_4bit:
                kwargs["load_in_4bit"] = True
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = {"": device.index if device.type == "cuda" else "cpu"}
            ref = AutoModelForCausalLM.from_pretrained(sft_dir, **kwargs)
            _maybe_resize_to_tokenizer_vocab(ref, name="ref(full)")
            ref.eval()
            for p in ref.parameters():
                p.requires_grad = False

    return policy, ref


# -------------------------
# args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Q-learning (HF/Qwen) for composition graphs")

    # data & model
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_paths_per_pair", type=int, default=20)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--sft_dir", type=str, required=True, help="SFT checkpoint dir (full model dir OR LoRA adapter dir).")
    p.add_argument("--trust_remote_code", action="store_true")

    # precision / loading
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")

    # lora
    p.add_argument("--lora_r", type=int, default=0)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # training
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.96)
    p.add_argument("--target_ema", type=float, default=0.0)

    # rollout
    p.add_argument("--max_rollout_steps", type=int, default=32)
    p.add_argument("--rollout_top_k", type=int, default=1)
    p.add_argument("--rollout_temp_start", type=float, default=0.0)
    p.add_argument("--rollout_temp_end", type=float, default=0.0)
    p.add_argument("--temp_warmup_iters", type=int, default=0)
    p.add_argument("--epsilon_start", type=float, default=0.04)
    p.add_argument("--epsilon_end", type=float, default=0.01)
    p.add_argument("--epsilon_warmup_iters", type=int, default=600)

    # IMPORTANT: make epsilon meaningful even if rollout_top_k=1
    p.add_argument("--epsilon_explore_top_k", type=int, default=20)

    # invalid handling
    p.add_argument("--allow_invalid_continue", action="store_true")
    p.add_argument("--max_invalid_transitions", type=int, default=2)

    # reward
    p.add_argument("--reward_type", choices=["process", "outcome"], default="process")
    p.add_argument("--reward_hit_target", type=float, default=1.5)
    p.add_argument("--reward_valid_transition", type=float, default=0.1)
    p.add_argument("--reward_stage_bridge", type=float, default=0.3)
    p.add_argument("--reward_stage_bridge_only_once", action="store_true")
    p.add_argument("--reward_invalid_transition", type=float, default=0.25)
    p.add_argument("--reward_invalid_token", type=float, default=1.0)
    p.add_argument("--reward_stop", type=float, default=-0.1)
    p.add_argument("--penalty_stage2_detour", type=float, default=0.2)
    p.add_argument("--penalty_stage3_detour", type=float, default=0.2)
    p.add_argument("--penalty_repeat_node", type=float, default=0.15)
    p.add_argument("--step_penalty", type=float, default=0.02)

    # node range
    p.add_argument("--node_min", type=int, default=0)
    p.add_argument("--node_max", type=int, default=149)

    # KL switch
    p.add_argument("--kl_coef", type=float, default=0.03)
    p.add_argument("--kl_warmup_iters", type=int, default=0)
    p.add_argument("--kl_anneal_iters", type=int, default=1000)
    p.add_argument("--kl_mode", choices=["action", "full"], default="action")

    # eval & save
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--max_eval_pairs", type=int, default=500)
    p.add_argument("--eval_temperature", type=float, default=1e-3)
    p.add_argument("--log_dir", type=str, default="out_ql_hf")

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

    data_dir = Path(args.data_dir).resolve()
    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"
    if not train_txt.exists():
        raise FileNotFoundError(train_txt)
    if not test_txt.exists():
        raise FileNotFoundError(test_txt)

    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    meta = pickle.load(open(meta_path, "rb"))
    block_size = int(meta.get("block_size", 63))

    # tokenizer: prefer sft_dir tokenizer if available
    sft_path = Path(args.sft_dir)
    tok_dir = sft_path if (sft_path / "tokenizer.json").exists() else (data_dir / "tokenizer")
    if not tok_dir.exists():
        raise FileNotFoundError(f"tokenizer dir not found: {tok_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    print(f"[tokenizer] loaded from {tok_dir}")

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("pad_token_id == eos_token_id (should not happen with your prepare script).")

    tokenizer_vocab_size = len(tokenizer)

    stage_info_path = data_dir / "stage_info.pkl"
    if not stage_info_path.exists():
        raise FileNotFoundError(f"stage_info.pkl not found: {stage_info_path}")
    stage_info = pickle.load(open(stage_info_path, "rb"))
    stages: List[List[int]] = stage_info["stages"]

    stage_sets = {
        "S1": set(stages[0]) if len(stages) > 0 else set(),
        "S2": set(stages[1]) if len(stages) > 1 else set(),
        "S3": set(stages[2]) if len(stages) > 2 else set(),
    }

    graph_path = data_dir / "composition_graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"composition_graph.graphml not found: {graph_path}")
    graph = nx.read_graphml(graph_path)

    out_dir = prepare_output_dir(args.log_dir)
    metrics_path = out_dir / "metrics_ql.jsonl"
    print(f"[out_dir] {out_dir}")
    print(
        f"[meta] block_size={block_size}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}, "
        f"tokenizer_vocab_size={tokenizer_vocab_size}"
    )

    policy_model, ref_model = load_policy_model_and_ref(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        device=device,
        args=args,
        vocab_size=tokenizer_vocab_size,
    )
    policy_model.train()

    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.eos_token_id = tokenizer.eos_token_id
    policy_model.config.use_cache = True  # rollout will use cache; training forward sets use_cache=False explicitly

    if ref_model is not None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.eos_token_id = tokenizer.eos_token_id
        ref_model.config.use_cache = False
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    target_model = None
    if args.target_ema and args.target_ema > 0:
        raise NotImplementedError("target_ema>0 is not implemented in this minimal HF version.")

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    train_pairs = load_pairs_unique(train_txt)
    eval_pairs: List[Pair] = []
    for line in test_txt.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            eval_pairs.append((int(parts[0]), int(parts[1])))

    print(f"[data] train unique pairs={len(train_pairs)}, eval pairs={len(eval_pairs)}")

    bucket_names = ["S1->S2", "S2->S3", "S1->S3"]

    t0_all = time.perf_counter()

    for iteration in range(1, args.max_iters + 1):
        t_iter = time.perf_counter()

        temperature = current_temperature(iteration, args)
        epsilon = current_epsilon(iteration, args)
        kl_coef_cur = current_kl_coef(iteration, args)

        batch_pairs = random.choices(train_pairs, k=args.batch_size)

        losses: List[Tensor] = []
        kl_values: List[float] = []
        td_errors: List[float] = []

        success_count = 0
        hit_count = 0
        invalid_edge_count = 0
        invalid_tok_count = 0
        stage2_visit_count = 0

        bucket_success_sum = defaultdict(float)
        bucket_counts = defaultdict(int)
        ep_rewards: List[float] = []

        for (s, t) in batch_pairs:
            b = bucket_for_pair(s, t, stages)

            traj = sample_trajectory_hf(
                model=policy_model,
                tokenizer=tokenizer,
                source=s,
                target=t,
                graph=graph,
                stages=stages,
                stage_sets=stage_sets,
                pair_bucket=b,
                args=args,
                device=device,
                block_size=block_size,
                temperature=temperature,
                epsilon=epsilon,
            )

            prompt_len = len(traj["prompt_ids"])

            qloss, q_sel, targets, _ = q_learning_loss_single_traj(
                model=policy_model,
                target_model=target_model,
                traj_ids=traj["traj_ids"],
                prompt_len=prompt_len,
                actions=traj["actions"],
                rewards=traj["rewards"],
                dones=traj["dones"],
                device=device,
                gamma=args.gamma,
            )

            loss_i = qloss
            kl_val = 0.0

            if kl_coef_cur > 0.0 and ref_model is not None:
                if args.kl_mode == "action":
                    kl = kl_loss_action_level(
                        policy_model=policy_model,
                        ref_model=ref_model,
                        traj_ids=traj["traj_ids"],
                        prompt_len=prompt_len,
                        actions=traj["actions"],
                        device=device,
                    )
                else:
                    kl = kl_loss_full_vocab(
                        policy_model=policy_model,
                        ref_model=ref_model,
                        traj_ids=traj["traj_ids"],
                        prompt_len=prompt_len,
                        actions_len=len(traj["actions"]),
                        device=device,
                    )
                loss_i = loss_i + kl_coef_cur * kl
                kl_val = float(kl.detach().item())

            losses.append(loss_i)
            kl_values.append(kl_val)
            td_errors.append(float((q_sel - targets).abs().mean().item()))

            success = bool(traj["success"])
            success_count += int(success)
            hit_count += int(traj["hit_target"])
            invalid_edge_count += int(traj["invalid_transition"])
            invalid_tok_count += int(traj["invalid_token"])
            stage2_visit_count += int(traj["visited_stage2"])

            if traj["bucket"]:
                bucket_success_sum[traj["bucket"]] += 1.0 if success else 0.0
                bucket_counts[traj["bucket"]] += 1

            ep_rewards.append(float(sum(traj["rewards"])) if traj["rewards"] else 0.0)

        total_loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        opt.step()

        iter_time = time.perf_counter() - t_iter

        if iteration % 50 == 0:
            record = {
                "iter": iteration,
                "iter_time_sec": float(iter_time),
                "elapsed_min": float((time.perf_counter() - t0_all) / 60.0),
                "loss": float(total_loss.item()),
                "td_error": float(np.mean(td_errors)) if td_errors else 0.0,
                "kl_loss": float(np.mean(kl_values)) if kl_values else 0.0,
                "kl_coef_current": float(kl_coef_cur),
                "temperature": float(temperature),
                "epsilon": float(epsilon),
                "epsilon_explore_top_k": int(args.epsilon_explore_top_k),
                "rollout_top_k": int(args.rollout_top_k),
                "success_rate": success_count / len(batch_pairs),
                "hit_rate": hit_count / len(batch_pairs),
                "stage2_visit_rate": stage2_visit_count / len(batch_pairs),
                "invalid_edge_rate": invalid_edge_count / len(batch_pairs),
                "invalid_tok_rate": invalid_tok_count / len(batch_pairs),
                "avg_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            }
            for bn in bucket_names:
                cnt = bucket_counts.get(bn, 0)
                record[f"train_success/{bn}"] = (bucket_success_sum.get(bn, 0.0) / cnt) if cnt else 0.0

            print(json.dumps(record, ensure_ascii=False))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            eval_res = evaluate_model(
                model=policy_model,
                tokenizer=tokenizer,
                pairs=eval_pairs,
                stages=stages,
                graph=graph,
                device=device,
                args=args,
                block_size=block_size,
                max_pairs=args.max_eval_pairs,
            )
            eval_record = {"iter": iteration, "eval": eval_res}
            print(json.dumps(eval_record, ensure_ascii=False))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_dir = out_dir / f"ckpt_ql_{iteration}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with open(ckpt_dir / "train_args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            with open(ckpt_dir / "data_meta.pkl", "wb") as f:
                pickle.dump(meta, f)
            print(f"[save] {ckpt_dir}")


if __name__ == "__main__":
    main()