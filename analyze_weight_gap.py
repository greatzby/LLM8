#!/usr/bin/env python3
"""
analyze_weight_gap.py
通用的weight gap分析脚本 - 可以分析任何checkpoint目录
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import networkx as nx
from tqdm import tqdm
from datetime import datetime
import argparse

try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: Cannot import 'model.py'")
    exit()

# ==================== 配置类 (修改版) ====================

class ModelConfig:
    """模型配置类"""
    # 关键修改：__init__ 方法现在需要 data_dir
    def __init__(self, checkpoint_dir, data_dir, model_name="Model"):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir  # 保存 data_dir
        self.model_name = model_name
        self.device = torch.device('cpu')
        
        # 模型参数 - 这些可以从checkpoint中读取，暂时保留
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 120 # 根据您的训练脚本，默认值是120
        
        # 从 meta.pkl 加载 vocab_size
        self.load_meta_info()
        
        # 加载节点分组和图结构
        self.load_stage_info()
        self.load_graph_structure()
    
    def load_meta_info(self):
        """从 meta.pkl 加载 vocab_size 和其他元信息"""
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            print(f"❌ WARNING: meta.pkl not found at {meta_path}. Using default vocab_size=92.")
            self.vocab_size = 92 # 回退到默认值
            return

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        print(f"  ✓ Loaded meta info: vocab_size={self.vocab_size}")


    def load_stage_info(self):
        """加载节点分组信息"""
        # 关键修改：使用 self.data_dir
        stage_info_path = os.path.join(self.data_dir, 'stage_info.pkl')
        
        if not os.path.exists(stage_info_path):
            print(f"❌ ERROR: stage_info.pkl not found at {stage_info_path}")
            raise FileNotFoundError(f"Required file not found: {stage_info_path}")
        
        with open(stage_info_path, 'rb') as f:
            stage_info = pickle.load(f)
        
        self.S1, self.S2, self.S3 = stage_info['stages']
        
        # 转换为集合
        self.S1_set = set(self.S1)
        self.S2_set = set(self.S2)
        self.S3_set = set(self.S3)
        
        # 创建节点到token的映射 (假设总节点数是90)
        total_nodes = len(self.S1) + len(self.S2) + len(self.S3)
        self.node_to_token = {node: node + 2 for node in range(total_nodes)}
        self.token_to_node = {token: node for node, token in self.node_to_token.items()}
        
        # S1, S2, S3的token索引
        self.S1_tokens = [self.node_to_token[n] for n in self.S1]
        self.S2_tokens = [self.node_to_token[n] for n in self.S2]
        self.S3_tokens = [self.node_to_token[n] for n in self.S3]
        
        print(f"  ✓ Loaded stage info: S1={len(self.S1)}, S2={len(self.S2)}, S3={len(self.S3)} nodes")
    
    def load_graph_structure(self):
        """加载图结构"""
        # 关键修改：使用 self.data_dir
        graph_path = os.path.join(self.data_dir, 'composition_graph.graphml')
        
        if not os.path.exists(graph_path):
            print(f"❌ ERROR: composition_graph.graphml not found at {graph_path}")
            raise FileNotFoundError(f"Required file not found: {graph_path}")
        
        G = nx.read_graphml(graph_path)
        
        # 确保节点是整数
        if isinstance(list(G.nodes())[0], str):
            self.G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        else:
            self.G = G
        
        print(f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # 创建邻接矩阵
        self.A_true = np.zeros((self.vocab_size, self.vocab_size))
        
        for edge in self.G.edges():
            if edge[0] in self.node_to_token and edge[1] in self.node_to_token:
                source_token = self.node_to_token[edge[0]]
                target_token = self.node_to_token[edge[1]]
                self.A_true[source_token, target_token] = 1
        
        # 统计各类型边
        s1_s2_edges = 0
        s2_s3_edges = 0
        s1_s3_edges = 0
        
        for edge in self.G.edges():
            source, target = edge[0], edge[1]
            if source in self.S1_set and target in self.S2_set:
                s1_s2_edges += 1
            elif source in self.S2_set and target in self.S3_set:
                s2_s3_edges += 1
            elif source in self.S1_set and target in self.S3_set:
                s1_s3_edges += 1
        
        print(f"  Edge statistics: S1→S2={s1_s2_edges}, S2→S3={s2_s3_edges}, S1→S3={s1_s3_edges}")
        
        self.edge_stats = {
            'S1->S2': s1_s2_edges,
            'S2->S3': s2_s3_edges,
            'S1->S3': s1_s3_edges
        }

# ==================== 核心分析函数 (修改版) ====================

# extract_W_M_prime 和 calculate_weight_gap 函数无需修改，保持原样

# ... (此处省略 extract_W_M_prime 和 calculate_weight_gap 的代码) ...
def extract_W_M_prime(checkpoint_path, config):
    """提取W'_M矩阵"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    model_args = checkpoint.get('model_args', {})
    if not model_args:
        # Fallback if model_args not in checkpoint
        model_args = {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size,
            'block_size': 512, # A reasonable default
            'dropout': 0.0,
            'bias': False
        }
    
    # Override vocab_size from our config, which is loaded from meta.pkl
    model_args['vocab_size'] = config.vocab_size
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config.device)
    
    # Use strict=False to handle potential mismatches if model architecture changes
    state_dict = checkpoint['model']
    # Filter out unexpected keys
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    W_M_prime = []
    with torch.no_grad():
        for i in range(config.vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=config.device))
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[:config.vocab_size])
    
    return np.array(W_M_prime)

def calculate_weight_gap(W_M_prime, config, path_type):
    """计算weight gap"""
    if path_type == 'S1->S2':
        source_tokens = config.S1_tokens
        target_tokens = config.S2_tokens
    elif path_type == 'S2->S3':
        source_tokens = config.S2_tokens
        target_tokens = config.S3_tokens
    elif path_type == 'S1->S3':
        source_tokens = config.S1_tokens
        target_tokens = config.S3_tokens
    else:
        return {} # Should not happen
    
    # Ensure all tokens are within the bounds of W_M_prime
    if not (max(source_tokens) < W_M_prime.shape[0] and max(target_tokens) < W_M_prime.shape[1]):
        print(f"Warning: Token index out of bounds for {path_type}. Skipping.")
        return {'edge': np.nan, 'non_edge': np.nan, 'gap': np.nan}

    W_sub = W_M_prime[np.ix_(source_tokens, target_tokens)]
    A_sub = config.A_true[np.ix_(source_tokens, target_tokens)]
    
    edge_mask = (A_sub == 1)
    non_edge_mask = (A_sub == 0)
    
    stats = {}
    
    if np.sum(edge_mask) > 0:
        stats['edge'] = np.mean(W_sub[edge_mask])
    else:
        stats['edge'] = 0 # Or np.nan if you prefer
    
    if np.sum(non_edge_mask) > 0:
        stats['non_edge'] = np.mean(W_sub[non_edge_mask])
    else:
        stats['non_edge'] = 0 # Or np.nan
    
    stats['gap'] = stats['edge'] - stats['non_edge']
    stats['num_edges'] = np.sum(edge_mask)
    stats['num_non_edges'] = np.sum(non_edge_mask)
    
    return stats


# 关键修改：analyze_checkpoint_dir 函数现在需要 data_dir
def analyze_checkpoint_dir(checkpoint_dir, data_dir, iterations=None, output_name=None):
    """分析checkpoint目录"""
    
    # 如果没有指定输出名称，从路径提取
    if output_name is None:
        output_name = os.path.basename(checkpoint_dir)
    
    print("\n" + "="*80)
    print(f"🔬 WEIGHT GAP ANALYSIS: {output_name}")
    print("="*80)
    
    # ... (其余代码保持不变) ...
    # 只是在初始化 ModelConfig 时，需要传入 data_dir
    
    # ...
    
    print(f"\n📋 Configuration:")
    print(f"  • Checkpoint directory: {checkpoint_dir}")
    print(f"  • Data directory:       {data_dir}") # 添加打印信息
    # ...

    # 关键修改：初始化 ModelConfig 时传入 data_dir
    config = ModelConfig(checkpoint_dir, data_dir, output_name)
    
    # ... (其余所有分析和绘图代码都无需修改，可以保持原样) ...

# ... (此处省略了绘图和报告生成的大段代码，因为它们不需要修改) ...
# 请将您原始脚本中的绘图和报告部分复制到这里
def analyze_checkpoint_dir(checkpoint_dir, data_dir, iterations=None, output_name=None):
    """分析checkpoint目录"""
    
    if output_name is None:
        output_name = os.path.basename(checkpoint_dir)
    
    print("\n" + "="*80)
    print(f"🔬 WEIGHT GAP ANALYSIS: {output_name}")
    print("="*80)
    
    if iterations is None:
        iterations = list(range(5000, 51000, 5000))
    
    save_dir = f'weight_gap_analysis_{output_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  • Checkpoint directory: {checkpoint_dir}")
    print(f"  • Data directory:       {data_dir}")
    print(f"  • Iterations: {iterations}")
    print(f"  • Output: {save_dir}/")
    
    if not os.path.exists(checkpoint_dir):
        print(f"\n❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    available = []
    for it in iterations:
        if os.path.exists(os.path.join(checkpoint_dir, f'ckpt_{it}.pt')):
            available.append(it)
    
    print(f"  • Found {len(available)}/{len(iterations)} checkpoints: {available}")
    
    if not available:
        print("❌ No checkpoints found!")
        return
    
    print("\n" + "="*60)
    print("Loading graph and meta structure...")
    print("="*60)
    
    config = ModelConfig(checkpoint_dir, data_dir, output_name)
    
    print("\n" + "="*60)
    print("Analyzing checkpoints...")
    print("="*60)
    
    results = {
        'S1->S2': {'edge': [], 'non_edge': [], 'gap': []},
        'S2->S3': {'edge': [], 'non_edge': [], 'gap': []},
        'S1->S3': {'edge': [], 'non_edge': [], 'gap': []}
    }
    
    for iteration in tqdm(iterations, desc="Processing"):
        ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{iteration}.pt')
        
        if not os.path.exists(ckpt_path):
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
            continue
        
        try:
            W_M_prime = extract_W_M_prime(ckpt_path, config)
            
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                stats = calculate_weight_gap(W_M_prime, config, path_type)
                results[path_type]['edge'].append(stats['edge'])
                results[path_type]['non_edge'].append(stats['non_edge'])
                results[path_type]['gap'].append(stats['gap'])
                
        except Exception as e:
            print(f"\n  ⚠️ Error at iteration {iteration}: {e}")
            for path_type in results:
                results[path_type]['edge'].append(np.nan)
                results[path_type]['non_edge'].append(np.nan)
                results[path_type]['gap'].append(np.nan)
    
    print("\n" + "="*60)
    print("Generating detailed plots...")
    print("="*60)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Weight Gap Analysis: {output_name}', fontsize=16, fontweight='bold')
    
    path_types = ['S1->S2', 'S2->S3', 'S1->S3']
    colors = {'edge': '#2E86AB', 'non_edge': '#A23B72', 'gap': '#F18F01'}
    
    for i, path_type in enumerate(path_types):
        ax = axes[i, 0]
        edge_weights = results[path_type]['edge']
        valid_iters = [it for it, w in zip(iterations, edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='o', color=colors['edge'],
                   linewidth=2, markersize=6, alpha=0.8)
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors['edge'])
        
        ax.set_title('Average Edge Weight' if i == 0 else '', fontsize=12)
        ax.set_ylabel(f'{path_type}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax = axes[i, 1]
        non_edge_weights = results[path_type]['non_edge']
        valid_iters = [it for it, w in zip(iterations, non_edge_weights) if not np.isnan(w)]
        valid_weights = [w for w in non_edge_weights if not np.isnan(w)]
        
        if valid_weights:
            ax.plot(valid_iters, valid_weights, marker='s', color=colors['non_edge'],
                   linewidth=2, markersize=6, alpha=0.8, linestyle='--')
            ax.annotate(f'{valid_weights[-1]:.3f}', 
                       xy=(valid_iters[-1], valid_weights[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors['non_edge'])
        
        ax.set_title('Average Non-Edge Weight' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax = axes[i, 2]
        gaps = results[path_type]['gap']
        valid_iters = [it for it, g in zip(iterations, gaps) if not np.isnan(g)]
        valid_gaps = [g for g in gaps if not np.isnan(g)]
        
        if valid_gaps:
            ax.plot(valid_iters, valid_gaps, marker='^', color=colors['gap'],
                   linewidth=2.5, markersize=7)
            
            final_gap = valid_gaps[-1]
            ax.annotate(f'{final_gap:.3f}', 
                      xy=(valid_iters[-1], final_gap),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=10, color=colors['gap'], fontweight='bold')
            
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) > 0, 
                           alpha=0.2, color='green', interpolate=True)
            ax.fill_between(valid_iters, 0, valid_gaps, 
                           where=np.array(valid_gaps) < 0, 
                           alpha=0.2, color='red', interpolate=True)
            
            if path_type == 'S2->S3':
                if min(valid_gaps) > 0:
                    ax.text(0.95, 0.05, '✅ Always positive', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                else:
                    ax.text(0.95, 0.05, '⚠️ Goes negative', 
                           transform=ax.transAxes, ha='right', va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_title('Weight Gap (Edge - Non-Edge)' if i == 0 else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for i in range(3):
        for j in range(3):
            axes[i, j].set_xlabel('Training Iterations' if i == 2 else '', fontsize=11)
            axes[i, j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    detail_path = os.path.join(save_dir, 'weight_gap_detailed.png')
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed plot saved to: {detail_path}")
    plt.close(fig) # Close the figure to free up memory

    # ... (Rest of the plotting and saving code remains the same)
    # ... (Please copy from your original script) ...

# ==================== main 函数 (修改版) ====================

def main():
    parser = argparse.ArgumentParser(description='Analyze weight gap from checkpoints')
    # 关键修改：添加 --data_dir 参数
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory (e.g., out/composition_...)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory used for training (e.g., data/simple_graph/...)')
    parser.add_argument('--name', type=str, default=None, help='Output name for the analysis report (optional)')
    parser.add_argument('--iterations', type=str, default=None, help='Iterations to analyze, comma-separated (e.g., "5000,10000,15000")')
    
    args = parser.parse_args()
    
    # 解析iterations
    if args.iterations:
        iterations = [int(x) for x in args.iterations.split(',')]
    else:
        iterations = None # Will use default range
    
    # 关键修改：调用 analyze_checkpoint_dir 时传入 data_dir
    analyze_checkpoint_dir(args.checkpoint_dir, args.data_dir, iterations, args.name)

if __name__ == "__main__":
    # 删除了硬编码的分析路径，让脚本完全由命令行参数驱动
    main()