#!/usr/bin/env python3
"""
analyze_weight_gapnew.py
通用的 weight gap 分析脚本 - 计算整体的 edge 与 non-edge weight gap
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
    print("❌ Error: Cannot import 'model.py'")
    exit()


# ==================== 配置类 ====================


class ModelConfig:
    """模型配置类"""

    def __init__(self, checkpoint_dir, data_dir, model_name="Model"):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = os.path.abspath(data_dir)
        self.model_name = model_name
        self.device = torch.device("cpu")

        # 这些参数可以从 checkpoint 中读取，这里保留默认值
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 120

        # 初始化容器
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

        # 加载必要文件
        self.load_meta_info()
        self.load_stage_info()
        self.load_graph_structure()

    # ---------- 工具函数 ----------

    def _preferred_match(self, matches):
        """返回最短路径的匹配结果"""
        if not matches:
            return None
        matches = sorted(matches, key=lambda p: (len(p.split(os.sep)), len(p)))
        return matches[0]

    def _find_file(self, filename, friendly_name):
        """
        在 data_dir 及其相邻目录中查找指定文件。
        会优先匹配路径层级较浅的文件。
        """
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

        # 递归搜索 data_dir
        recursive_matches = glob.glob(
            os.path.join(self.data_dir, "**", filename), recursive=True
        )
        match = self._preferred_match(recursive_matches)
        if match and os.path.isfile(match):
            print(f"  ✓ Located {friendly_name} via recursive search at: {match}")
            return match

        print(f"❌ ERROR: Unable to locate {friendly_name} ({filename})")
        return None

    # ---------- 加载元信息 ----------

    def load_meta_info(self):
        """从 meta.pkl 加载 vocab_size 等信息"""
        meta_path = self._find_file("meta.pkl", "meta info")
        if meta_path is None:
            print(
                "❌ WARNING: meta.pkl not found. Falling back to default vocab_size=92."
            )
            self.vocab_size = 92
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.vocab_size = int(meta.get("vocab_size", 92))
        print(f"  ✓ Loaded meta info: vocab_size={self.vocab_size}")

    def load_stage_info(self):
        """加载节点分组信息"""
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
        """加载图结构并构建邻接矩阵"""
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


# ==================== 核心分析函数 ====================


def extract_W_M_prime(checkpoint_path, config):
    """提取 W'_M 矩阵"""
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
    for key, value in list(state_dict.items()):
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


def calculate_weight_gap(W_M_prime, config):
    """计算整体 weight gap（edge - non-edge）"""
    node_tokens = config.node_tokens
    if not node_tokens:
        raise ValueError("Node tokens list is empty. Check stage_info.pkl contents.")

    W_sub = W_M_prime[np.ix_(node_tokens, node_tokens)]
    A_sub = config.A_true[np.ix_(node_tokens, node_tokens)]

    edge_mask = A_sub == 1.0
    diag_mask = np.eye(len(node_tokens), dtype=bool)
    non_edge_mask = (A_sub == 0.0) & (~diag_mask)

    stats = {
        "edge": np.nan,
        "non_edge": np.nan,
        "gap": np.nan,
        "num_edges": int(edge_mask.sum()),
        "num_non_edges": int(non_edge_mask.sum()),
    }

    if stats["num_edges"] > 0:
        stats["edge"] = float(np.mean(W_sub[edge_mask]))
    if stats["num_non_edges"] > 0:
        stats["non_edge"] = float(np.mean(W_sub[non_edge_mask]))

    if not np.isnan(stats["edge"]) and not np.isnan(stats["non_edge"]):
        stats["gap"] = stats["edge"] - stats["non_edge"]

    return stats


def analyze_checkpoint_dir(checkpoint_dir, data_dir, iterations=None, output_name=None):
    """分析 checkpoint 目录下的整体 weight gap"""
    if output_name is None:
        output_name = os.path.basename(os.path.abspath(checkpoint_dir))

    print("\n" + "=" * 80)
    print(f"🔬 WEIGHT GAP ANALYSIS: {output_name}")
    print("=" * 80)

    if iterations is None:
        iterations = list(range(5000, 51000, 5000))

    save_dir = f"weight_gap_analysis_{output_name}"
    os.makedirs(save_dir, exist_ok=True)

    print("\n📋 Configuration:")
    print(f"  • Checkpoint directory: {checkpoint_dir}")
    print(f"  • Data directory:       {data_dir}")
    print(f"  • Iterations:           {iterations}")
    print(f"  • Output:               {save_dir}/")

    if not os.path.isdir(checkpoint_dir):
        print(f"\n❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return

    available_iterations = [
        it for it in iterations if os.path.isfile(os.path.join(checkpoint_dir, f"ckpt_{it}.pt"))
    ]
    print(
        f"  • Found {len(available_iterations)}/{len(iterations)} checkpoints: {available_iterations}"
    )

    if not available_iterations:
        print("❌ No checkpoints found! Aborting analysis.")
        return

    print("\n" + "=" * 60)
    print("Loading graph and meta structure...")
    print("=" * 60)
    config = ModelConfig(checkpoint_dir, data_dir, output_name)

    print("\n" + "=" * 60)
    print("Analyzing checkpoints...")
    print("=" * 60)

    results = {"iteration": [], "edge": [], "non_edge": [], "gap": []}

    for iteration in tqdm(iterations, desc="Processing"):
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{iteration}.pt")
        results["iteration"].append(iteration)

        if not os.path.isfile(ckpt_path):
            results["edge"].append(np.nan)
            results["non_edge"].append(np.nan)
            results["gap"].append(np.nan)
            continue

        try:
            W_M_prime = extract_W_M_prime(ckpt_path, config)
            stats = calculate_weight_gap(W_M_prime, config)
            results["edge"].append(stats["edge"])
            results["non_edge"].append(stats["non_edge"])
            results["gap"].append(stats["gap"])
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  ⚠️ Error at iteration {iteration}: {exc}")
            results["edge"].append(np.nan)
            results["non_edge"].append(np.nan)
            results["gap"].append(np.nan)

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Weight Gap Analysis (Overall): {output_name}", fontsize=16, fontweight="bold")

    metric_info = [
        ("edge", "Average Edge Weight", "#2E86AB", "o"),
        ("non_edge", "Average Non-Edge Weight", "#A23B72", "s"),
        ("gap", "Weight Gap (Edge - Non-Edge)", "#F18F01", "^"),
    ]

    for ax, (key, title, color, marker) in zip(axes, metric_info):
        values = results[key]
        iterations_array = results["iteration"]
        valid_pairs = [(it, val) for it, val in zip(iterations_array, values) if not np.isnan(val)]

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training Iterations", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

        if valid_pairs:
            xs, ys = zip(*valid_pairs)
            ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=6)
            ax.annotate(
                f"{ys[-1]:.4f}",
                xy=(xs[-1], ys[-1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=10,
                color=color,
                fontweight="bold" if key == "gap" else "normal",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
                alpha=0.7,
            )

        if key == "gap":
            ys_arr = np.array(values, dtype=float)
            xs_arr = np.array(iterations_array, dtype=float)
            mask = ~np.isnan(ys_arr)
            if mask.any():
                ax.fill_between(
                    xs_arr[mask],
                    0,
                    ys_arr[mask],
                    where=ys_arr[mask] >= 0,
                    interpolate=True,
                    alpha=0.2,
                    color="green",
                    label="Positive gap",
                )
                ax.fill_between(
                    xs_arr[mask],
                    0,
                    ys_arr[mask],
                    where=ys_arr[mask] < 0,
                    interpolate=True,
                    alpha=0.2,
                    color="red",
                    label="Negative gap",
                )
                ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(save_dir, "weight_gap_overall.png")
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Plot saved to: {plot_path}")

    # ---------- 保存数值结果 ----------
    summary_txt_path = os.path.join(save_dir, "weight_gap_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Weight Gap Analysis Summary ({output_name})\n")
        f.write(f"Generated at: {now}\n")
        f.write(f"Checkpoint directory: {checkpoint_dir}\n")
        f.write(f"Data directory:       {data_dir}\n")
        f.write(f"Stages: S1={len(config.S1)}, S2={len(config.S2)}, S3={len(config.S3)}\n")
        f.write(
            f"Graph edges: S1→S2={config.edge_stats['S1->S2']}, "
            f"S2→S3={config.edge_stats['S2->S3']}, S1→S3={config.edge_stats['S1->S3']}\n"
        )
        f.write("\nIterations analyzed:\n")
        for it, edge_val, non_edge_val, gap_val in zip(
            results["iteration"], results["edge"], results["non_edge"], results["gap"]
        ):
            f.write(
                f"  Iter {it:>6}: "
                f"edge={edge_val:>10.6f}  "
                f"non_edge={non_edge_val:>10.6f}  "
                f"gap={gap_val:>10.6f}\n"
                if not np.isnan(edge_val)
                else f"  Iter {it:>6}: (missing checkpoint)\n"
            )

        edge_values = np.array(results["edge"], dtype=float)
        non_edge_values = np.array(results["non_edge"], dtype=float)
        gap_values = np.array(results["gap"], dtype=float)

        def safe_stats(arr):
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan, np.nan, np.nan
            return float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))

        edge_min, edge_max, edge_mean = safe_stats(edge_values)
        non_edge_min, non_edge_max, non_edge_mean = safe_stats(non_edge_values)
        gap_min, gap_max, gap_mean = safe_stats(gap_values)

        f.write("\nSummary statistics (excluding NaNs):\n")
        f.write(
            f"  Edge     : min={edge_min:.6f}, max={edge_max:.6f}, mean={edge_mean:.6f}\n"
            if not np.isnan(edge_min)
            else "  Edge     : no valid data\n"
        )
        f.write(
            f"  Non-Edge : min={non_edge_min:.6f}, max={non_edge_max:.6f}, mean={non_edge_mean:.6f}\n"
            if not np.isnan(non_edge_min)
            else "  Non-Edge : no valid data\n"
        )
        f.write(
            f"  Gap      : min={gap_min:.6f}, max={gap_max:.6f}, mean={gap_mean:.6f}\n"
            if not np.isnan(gap_min)
            else "  Gap      : no valid data\n"
        )

    print(f"✅ Summary saved to: {summary_txt_path}")

    print("\n🎯 Analysis complete.")


# ==================== main 函数 ====================


def main():
    parser = argparse.ArgumentParser(description="Analyze overall weight gap from checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., out/composition_...)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory used for training (contains meta/stage info)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional output name for the analysis report",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help='Comma-separated iterations to analyze (e.g., "5000,10000,15000")',
    )

    args = parser.parse_args()

    if args.iterations:
        iterations = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
    else:
        iterations = None

    analyze_checkpoint_dir(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        iterations=iterations,
        output_name=args.name,
    )


if __name__ == "__main__":
    main()