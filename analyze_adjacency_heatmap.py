#!/usr/bin/env python3
"""
analyze_adjacency_heatmap.py

Given a checkpoint (usually the last iteration), compute the W_M matrix
as defined in the ALPINE paper, compare it against the ground-truth adjacency
matrix A_true, and visualize both as heatmaps (optionally also the difference).

This script reuses the configuration/loading logic from the weight-gap script.
"""

import argparse
import glob
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

try:
    from model import GPTConfig, GPT
except ImportError:
    raise ImportError("❌ Error: Cannot import 'model.py'. Please ensure it is available.") from None


# ==================== 配置类（与 weight gap 脚本一致） ====================


class ModelConfig:
    """模型配置类"""

    def __init__(self, checkpoint_dir, data_dir, model_name="Model"):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = os.path.abspath(data_dir)
        self.model_name = model_name
        self.device = torch.device("cpu")

        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 120

        self.vocab_size = None
        self.S1 = self.S2 = self.S3 = []
        self.S1_set = self.S2_set = self.S3_set = set()
        self.S1_tokens = self.S2_tokens = self.S3_tokens = []
        self.node_to_token = {}
        self.token_to_node = {}
        self.node_tokens = []
        self.G = None
        self.A_true = None
        self.edge_stats = {}

        self.load_meta_info()
        self.load_stage_info()
        self.load_graph_structure()

    def _preferred_match(self, matches):
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: (len(p.split(os.sep)), len(p)))
        return matches[0]

    def _find_file(self, filename, friendly_name):
        search_dirs = [
            self.data_dir,
            os.path.dirname(self.data_dir),
            os.path.join(self.data_dir, "meta"),
            os.path.join(self.data_dir, "data"),
            os.path.join(self.data_dir, "graph"),
            os.path.join(self.data_dir, "graphs"),
            os.path.join(os.path.dirname(self.data_dir), "data"),
            os.path.join(os.path.dirname(self.data_dir), "graphs"),
        ]

        checked = set()
        for directory in search_dirs:
            directory = os.path.abspath(directory)
            if directory in checked or not os.path.isdir(directory):
                continue
            checked.add(directory)
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                print(f"  ✓ Located {friendly_name} at: {candidate}")
                return candidate

        recursive_matches = glob.glob(
            os.path.join(self.data_dir, "**", filename), recursive=True
        )
        match = self._preferred_match(recursive_matches)
        if match and os.path.isfile(match):
            print(f"  ✓ Located {friendly_name} via recursive search at: {match}")
            return match

        print(f"❌ ERROR: Unable to locate {friendly_name} ({filename})")
        return None

    def load_meta_info(self):
        meta_path = self._find_file("meta.pkl", "meta info")
        if meta_path is None:
            print("❌ WARNING: meta.pkl not found. Falling back to default vocab_size=92.")
            self.vocab_size = 92
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.vocab_size = int(meta.get("vocab_size", 92))
        print(f"  ✓ Loaded meta info: vocab_size={self.vocab_size}")

    def load_stage_info(self):
        stage_info_path = self._find_file("stage_info.pkl", "stage info")
        if stage_info_path is None:
            raise FileNotFoundError(
                f"Required file not found: stage_info.pkl (search base: {self.data_dir})"
            )

        with open(stage_info_path, "rb") as f:
            stage_info = pickle.load(f)

        self.S1, self.S2, self.S3 = stage_info["stages"]

        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)

        total_nodes = len(self.S1) + len(self.S2) + len(self.S3)
        self.node_to_token = {node: node + 2 for node in range(total_nodes)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}

        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]

        self.node_tokens = sorted(
            set(self.S1_tokens + self.S2_tokens + self.S3_tokens)
        )

        print(
            f"  ✓ Loaded stage info: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)} nodes"
        )

    def load_graph_structure(self):
        graph_path = self._find_file("composition_graph.graphml", "graph structure")
        if graph_path is None:
            raise FileNotFoundError(
                f"Required file not found: composition_graph.graphml (search base: {self.data_dir})"
            )

        G = nx.read_graphml(graph_path)
        if isinstance(next(iter(G.nodes())), str):
            self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        else:
            self.G = G

        print(
            f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges"
        )

        self.A_true = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

        s1_s2_edges = 0
        s2_s3_edges = 0
        s1_s3_edges = 0

        for source, target in self.G.edges():
            if (
                source in self.node_to_token
                and target in self.node_to_token
                and self.node_to_token[source] < self.vocab_size
                and self.node_to_token[target] < self.vocab_size
            ):
                src_token = self.node_to_token[source]
                tgt_token = self.node_to_token[target]
                self.A_true[src_token, tgt_token] = 1.0

            if source in self.S1_set and target in self.S2_set:
                s1_s2_edges += 1
            elif source in self.S2_set and target in self.S3_set:
                s2_s3_edges += 1
            elif source in self.S1_set and target in self.S3_set:
                s1_s3_edges += 1

        self.edge_stats = {
            "S1->S2": s1_s2_edges,
            "S2->S3": s2_s3_edges,
            "S1->S3": s1_s3_edges,
        }

        print(
            f"  Edge statistics: S1→S2={s1_s2_edges}, "
            f"S2→S3={s2_s3_edges}, S1→S3={s1_s3_edges}"
        )


# ==================== 核心函数（重用 weight gap 脚本） ====================


def extract_W_M_prime(checkpoint_path, config):
    """提取 W_M 矩阵"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    model_args = checkpoint.get("model_args", {})
    if not model_args:
        model_args = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "vocab_size": config.vocab_size,
            "block_size": 512,
            "dropout": 0.0,
            "bias": False,
        }

    model_args["vocab_size"] = config.vocab_size

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    W_M_prime = []
    with torch.no_grad():
        for token_idx in range(config.vocab_size):
            token = torch.tensor([token_idx], device=config.device)
            token_emb = model.transformer.wte(token)
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[: config.vocab_size])

    return np.array(W_M_prime, dtype=np.float32)


# ==================== 辅助函数 ====================


def locate_checkpoint(checkpoint_dir, iteration=None, checkpoint_path=None):
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        filename = os.path.basename(checkpoint_path)
        iter_hint = "".join([c for c in filename if c.isdigit()])
        iteration = int(iter_hint) if iter_hint else None
        return checkpoint_path, iteration

    if iteration is not None:
        candidate = os.path.join(checkpoint_dir, f"ckpt_{iteration}.pt")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Checkpoint ckpt_{iteration}.pt not found in {checkpoint_dir}")
        return candidate, iteration

    pattern = os.path.join(checkpoint_dir, "ckpt_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    def extract_iter(path):
        name = os.path.basename(path)
        digits = "".join([c for c in name if c.isdigit()])
        return int(digits) if digits else -1

    matches.sort(key=lambda p: extract_iter(p))
    chosen = matches[-1]
    return chosen, extract_iter(chosen)


def compute_metrics(W_sub, A_sub, threshold=0.0):
    mask = np.ones_like(A_sub, dtype=bool)
    np.fill_diagonal(mask, False)

    A_vec = A_sub[mask].astype(float)
    W_vec = W_sub[mask].astype(float)

    diff = W_vec - A_vec
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))

    if np.std(A_vec) > 0 and np.std(W_vec) > 0:
        corr = float(np.corrcoef(A_vec, W_vec)[0, 1])
    else:
        corr = float("nan")

    W_binary = (W_vec >= threshold).astype(int)
    A_binary = A_vec.astype(int)

    tp = int(np.sum((W_binary == 1) & (A_binary == 1)))
    fp = int(np.sum((W_binary == 1) & (A_binary == 0)))
    fn = int(np.sum((W_binary == 0) & (A_binary == 1)))
    tn = int(np.sum((W_binary == 0) & (A_binary == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision is not None and recall is not None and not np.isnan(precision) and not np.isnan(recall):
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    else:
        f1 = float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "corr": corr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


def plot_heatmaps(A_sub, W_sub, diff_sub, node_tokens, output_path, title=None, include_diff=False):
    num_cols = 3 if include_diff else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
    if num_cols == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    im0 = axes[0].imshow(A_sub, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth $A^{true}$")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmin_w = np.percentile(W_sub, 1)
    vmax_w = np.percentile(W_sub, 99)
    if vmin_w == vmax_w:
        vmin_w -= 1
        vmax_w += 1
    im1 = axes[1].imshow(W_sub, cmap="viridis", vmin=vmin_w, vmax=vmax_w)
    axes[1].set_title("$W_M$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if include_diff:
        max_abs = np.max(np.abs(diff_sub))
        if max_abs == 0:
            max_abs = 1.0
        im2 = axes[2].imshow(diff_sub, cmap="seismic", vmin=-max_abs, vmax=max_abs)
        axes[2].set_title("$W_M - A^{true}$")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Target node token index")
        ax.set_ylabel("Source node token index")
        ax.set_xticks(range(len(node_tokens)))
        ax.set_yticks(range(len(node_tokens)))
        ax.set_xticklabels(node_tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(node_tokens, fontsize=8)
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Heatmap saved to: {output_path}")


def save_metrics(metrics, output_path, checkpoint_path, iteration, config):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Adjacency Comparison Summary\n")
        f.write(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Iteration : {iteration}\n")
        f.write(f"Data dir  : {config.data_dir}\n")
        f.write("\n")
        f.write(f"Threshold : {metrics['threshold']:.4f}\n")
        f.write(f"MSE       : {metrics['mse']:.6f}\n")
        f.write(f"MAE       : {metrics['mae']:.6f}\n")
        f.write(f"Max |Δ|   : {metrics['max_abs']:.6f}\n")
        f.write(f"Corr      : {metrics['corr']:.6f}\n")
        f.write("\n")
        f.write(f"TP        : {metrics['tp']}\n")
        f.write(f"FP        : {metrics['fp']}\n")
        f.write(f"FN        : {metrics['fn']}\n")
        f.write(f"TN        : {metrics['tn']}\n")
        f.write(f"Precision : {metrics['precision']:.6f}\n")
        f.write(f"Recall    : {metrics['recall']:.6f}\n")
        f.write(f"F1        : {metrics['f1']:.6f}\n")
    print(f"✅ Metrics saved to: {output_path}")


# ==================== 主流程 ====================


def main():
    parser = argparse.ArgumentParser(description="Visualize W_M vs ground-truth adjacency at a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing ckpt_*.pt files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with meta.pkl, stage_info.pkl, etc.")
    parser.add_argument("--iteration", type=int, default=None, help="Iteration number (e.g., 50000). If omitted, use the largest one.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional explicit path to checkpoint file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store outputs. Default: heatmap_<name>/")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for binarizing W_M when computing precision/recall/F1.")
    parser.add_argument("--save_matrix", action="store_true", help="Save W_M, A_true, and difference as .npy files.")
    parser.add_argument("--include_diff", action="store_true", help="Also plot the heatmap of W_M - A_true.")
    parser.add_argument("--fig_title", type=str, default=None, help="Custom title for the heatmap figure.")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    checkpoint_path, iteration = locate_checkpoint(
        args.checkpoint_dir,
        iteration=args.iteration,
        checkpoint_path=args.checkpoint_path,
    )
    print(f"\n🎯 Using checkpoint: {checkpoint_path} (iteration={iteration})\n")

    print("=" * 60)
    print("Loading configuration / graph info...")
    print("=" * 60)
    config = ModelConfig(args.checkpoint_dir, args.data_dir, model_name=os.path.basename(args.checkpoint_dir))

    print("\n=" * 30)
    print("Extracting W_M matrix...")
    print("=" * 60)
    W_M = extract_W_M_prime(checkpoint_path, config)

    node_tokens = config.node_tokens
    if not node_tokens:
        raise ValueError("Node tokens list is empty. Check stage_info.pkl contents.")

    A_sub = config.A_true[np.ix_(node_tokens, node_tokens)]
    W_sub = W_M[np.ix_(node_tokens, node_tokens)]
    diff_sub = W_sub - A_sub

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"adjacency_heatmap_{os.path.basename(os.path.abspath(args.checkpoint_dir))}"
    os.makedirs(output_dir, exist_ok=True)

    if args.fig_title:
        fig_title = args.fig_title
    else:
        fig_title = f"{config.model_name} — Iter {iteration}"

    plot_path = os.path.join(output_dir, f"adjacency_heatmap_iter_{iteration}.png")
    plot_heatmaps(
        A_sub,
        W_sub,
        diff_sub,
        node_tokens,
        plot_path,
        title=fig_title,
        include_diff=args.include_diff,
    )

    metrics = compute_metrics(W_sub, A_sub, threshold=args.threshold)
    metrics_path = os.path.join(output_dir, f"adjacency_metrics_iter_{iteration}.txt")
    save_metrics(metrics, metrics_path, checkpoint_path, iteration, config)

    if args.save_matrix:
        np.save(os.path.join(output_dir, f"W_M_iter_{iteration}.npy"), W_sub)
        np.save(os.path.join(output_dir, f"A_true_nodes.npy"), A_sub)
        np.save(os.path.join(output_dir, f"W_minus_A_iter_{iteration}.npy"), diff_sub)
        print("✅ Saved raw matrices (.npy).")

    print("\n📊 Summary metrics:")
    for key in ["mse", "mae", "max_abs", "corr", "precision", "recall", "f1"]:
        print(f"  {key:>9}: {metrics[key]:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()