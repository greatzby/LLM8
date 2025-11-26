#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphA Mechanism Analysis Script (Final Version)
整合了：
1. Atomic Confidence (原子置信度 - 解释为何能泛化/不迷路)
2. Decomposition Paradox (分解悖论 - 解释为何无法停止)
3. Procrustes Alignment (几何漂移 - 解释表征重组)
4. Effective Dimension (维度分析 - 解释表征压缩)
5. Weight Analysis (权重分析 - 解释参数变化)

Usage:
    python analyze_mechanism_final.py \
        --data_dir data/datasets/graphA_pg030_tier3_P13_0 \
        --ckpt_sft out/composition_20251022_124024/ckpt_50000.pt \
        --ckpt_pg out_pg/pg_20251117_020015/ckpt_pg_18000.pt \
        --ckpt_ql out_ql_no_target/ql_20251119_151554/ckpt_ql_20000.pt \
        --output_dir analysis_results_final
"""

import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from collections import defaultdict
from model import GPT, GPTConfig

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_sft", type=str, required=True)
    parser.add_argument("--ckpt_pg", type=str, required=True)
    parser.add_argument("--ckpt_ql", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_results_final")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=500)
    return parser.parse_args()

def load_model(ckpt_path, device, meta_vocab_size):
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_args['vocab_size'] = meta_vocab_size
    model_args['dropout'] = 0.0 # 确保确定性
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def get_s1_s3_samples(test_file, stages, graph, num_samples=500):
    """获取合法的 S1->S3 样本"""
    S1, S2, S3 = stages[0], stages[1], stages[2]
    samples = []
    with open(test_file, 'r') as f:
        lines = f.readlines()
    import random
    random.seed(42)
    random.shuffle(lines)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue
        src, tgt = int(parts[0]), int(parts[1])
        if src in S1 and tgt in S3:
            valid_s2 = []
            for s2_node in S2:
                if graph.has_edge(str(src), str(s2_node)) and graph.has_edge(str(s2_node), str(tgt)):
                    valid_s2.append(s2_node)
            if valid_s2:
                # 随机选一个合法的中间点作为 Ground Truth
                samples.append({'src': src, 'tgt': tgt, 's2_step': valid_s2[0]})
        if len(samples) >= num_samples: break
    return samples

# --- Analysis 1: Atomic Confidence (Why it generalizes?) ---
def analyze_atomic_confidence(models, samples, stoi, device, output_dir):
    """
    分析在 S1 节点时，模型对正确下一步 (S2) 的置信度。
    Input: [src, tgt, src]
    Expected Output: s2_step
    """
    print("Running Atomic Confidence Analysis...")
    results = []
    
    for sample in samples:
        context = [stoi[str(sample['src'])], stoi[str(sample['tgt'])], stoi[str(sample['src'])]]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        correct_next = stoi[str(sample['s2_step'])]
        
        for name, model in models.items():
            with torch.no_grad():
                logits, _ = model(x)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            p_correct = probs[correct_next].item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            results.append({
                'Model': name,
                'Prob_Correct_Next': p_correct,
                'Entropy': entropy
            })
            
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confidence
    sns.violinplot(data=df, x='Model', y='Prob_Correct_Next', hue='Model', ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Atomic Confidence: P(S2 | S1, Goal=S3)")
    axes[0].set_ylabel("Probability of Correct Next Step")
    
    # 2. Entropy (Sharpening)
    sns.violinplot(data=df, x='Model', y='Entropy', hue='Model', ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Decision Uncertainty (Entropy)")
    axes[1].set_ylabel("Entropy (Lower = Sharper Rules)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atomic_confidence.png'))
    plt.close()
    print("Atomic Confidence analysis saved.")

# --- Analysis 2: Decomposition Paradox (Why it fails sub-tasks?) ---
def analyze_decomposition_paradox(models, samples, stoi, itos, stages, device, output_dir):
    """
    分析悖论：当我们要求 S1->S2 时，模型在到达 S2 后，是停止(EOS)还是继续(S3)?
    注意：我们这里用 S1->S3 的样本，但构造 S1->S2 的 Prompt。
    Prompt: [src, s2_step, src, s2_step] -> 此时应该输出 EOS
    """
    print("Running Decomposition Paradox Analysis...")
    S3_nodes = stages[2]
    stop_token_id = stoi['\n']
    results = []
    
    # 我们只需要一部分样本来展示这个现象
    subset_samples = samples[:100]
    
    for sample in subset_samples:
        # 构造一个 "假装" 任务是 S1->S2 的场景
        # Prompt: src(S1) -> tgt(S2) -> src(S1) -> s2_step(S2)
        # 此时模型应该输出 EOS
        s2_node = sample['s2_step']
        context = [
            stoi[str(sample['src'])], 
            stoi[str(s2_node)],  # 目标设为 S2
            stoi[str(sample['src'])], 
            stoi[str(s2_node)]   # 已经到达 S2
        ]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        
        for name, model in models.items():
            with torch.no_grad():
                logits, _ = model(x)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            p_eos = probs[stop_token_id].item()
            
            # 计算所有 S3 节点的概率总和 (模型想继续走的概率)
            p_continue_s3 = 0.0
            for node in S3_nodes:
                if str(node) in stoi:
                    p_continue_s3 += probs[stoi[str(node)]].item()
            
            results.append({
                'Model': name,
                'Probability': p_eos,
                'Token Type': 'EOS (Stop)'
            })
            results.append({
                'Model': name,
                'Probability': p_continue_s3,
                'Token Type': 'Any S3 Node (Continue)'
            })

    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Probability', hue='Token Type', palette="viridis", errorbar=None)
    plt.title("Decomposition Paradox: At Node S2, does it Stop or Continue?")
    plt.ylabel("Average Probability")
    plt.yscale('log') # 对数坐标，因为差异可能很大
    
    plt.savefig(os.path.join(output_dir, 'decomposition_paradox.png'))
    plt.close()
    print("Decomposition Paradox analysis saved.")

# --- Analysis 3: Procrustes & Effective Dimension (Geometry) ---
def analyze_geometry(models, samples, stoi, device, output_dir):
    print("Running Geometry Analysis (Procrustes & PCA)...")
    
    # 提取 S2 节点处的 Hidden States
    hidden_states = defaultdict(list)
    for sample in samples:
        # Input: [src, tgt, src, s2] -> 看最后一个 token 的输出
        context = [stoi[str(sample['src'])], stoi[str(sample['tgt'])], stoi[str(sample['src'])], stoi[str(sample['s2_step'])]]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        
        for name, model in models.items():
            features = {}
            def get_activation(name):
                def hook(model, input, output):
                    features[name] = output.detach()
                return hook
            # Hook ln_f (final layer norm)
            handle = model.transformer.ln_f.register_forward_hook(get_activation('ln_f'))
            model(x)
            handle.remove()
            hidden_states[name].append(features['ln_f'][0, -1, :].cpu().numpy())

    m_sft = np.array(hidden_states['SFT'])
    m_pg = np.array(hidden_states['PG'])
    m_ql = np.array(hidden_states['QL'])
    
    # 1. Procrustes Alignment (Drift)
    def compute_residual(m1, m2):
        if m1.shape != m2.shape: return None
        m1_std, m2_aligned, disparity = procrustes(m1, m2)
        # 逐样本计算欧氏距离
        return np.linalg.norm(m1_std - m2_aligned, axis=1)

    res_pg = compute_residual(m_sft, m_pg)
    res_ql = compute_residual(m_sft, m_ql)
    
    plot_data = []
    if res_pg is not None:
        for v in res_pg: plot_data.append({'Comparison': 'SFT vs PG', 'Residual Distance': v})
    if res_ql is not None:
        for v in res_ql: plot_data.append({'Comparison': 'SFT vs QL', 'Residual Distance': v})
        
    if plot_data:
        df_proc = pd.DataFrame(plot_data)
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df_proc, x='Residual Distance', hue='Comparison', kde=True, element="step")
        plt.title("Geometric Drift after Alignment (Representation Restructuring)")
        plt.savefig(os.path.join(output_dir, 'procrustes_drift.png'))
        plt.close()
    
    # 2. Effective Dimension (PCA)
    plt.figure(figsize=(10, 6))
    for matrix, name in zip([m_sft, m_pg, m_ql], ['SFT', 'PG', 'QL']):
        if len(matrix) == 0: continue
        matrix = matrix - np.mean(matrix, axis=0) # Center
        pca = PCA()
        pca.fit(matrix)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        # 找到 90% 方差的截断点
        k90 = np.argmax(cumsum > 0.9) + 1
        plt.plot(cumsum, label=f'{name} (90% var @ k={k90})', linewidth=2)
        
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Effective Dimensionality of Hidden States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'effective_dimension.png'))
    plt.close()
    print("Geometry analysis saved.")

# --- Analysis 4: Weight Change (Mechanism) ---
def analyze_weights(models, output_dir):
    print("Running Weight Analysis...")
    base_model = models['SFT']
    diff_results = []
    
    for name in ['PG', 'QL']:
        target_model = models[name]
        for (k1, v1), (k2, v2) in zip(base_model.named_parameters(), target_model.named_parameters()):
            if v1.shape != v2.shape: continue
            
            diff = (v1 - v2).norm().item()
            rel_diff = diff / (v1.norm().item() + 1e-9)
            
            layer_type = "Other"
            if "wte" in k1: layer_type = "Embedding"
            elif "attn" in k1: layer_type = "Attention"
            elif "mlp" in k1: layer_type = "MLP"
            elif "ln" in k1: layer_type = "LayerNorm"
            
            diff_results.append({
                'Method': name,
                'Layer Type': layer_type,
                'Relative Diff': rel_diff
            })
            
    df = pd.DataFrame(diff_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Layer Type', y='Relative Diff', hue='Method', errorbar=None)
    plt.title("Relative Weight Change by Module Type")
    plt.ylabel("Relative L2 Norm Difference")
    plt.savefig(os.path.join(output_dir, 'weight_change.png'))
    plt.close()
    print("Weight analysis saved.")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Meta
    meta_path = os.path.join(args.data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f: meta = pickle.load(f)
    stoi, vocab_size = meta['stoi'], meta['vocab_size']
    itos = meta['itos']
    
    stage_path = os.path.join(args.data_dir, 'stage_info.pkl')
    with open(stage_path, 'rb') as f: stages = pickle.load(f)['stages']
    
    G = nx.read_graphml(os.path.join(args.data_dir, 'composition_graph.graphml'))
    
    # Load Models
    models = {
        'SFT': load_model(args.ckpt_sft, args.device, vocab_size),
        'PG': load_model(args.ckpt_pg, args.device, vocab_size),
        'QL': load_model(args.ckpt_ql, args.device, vocab_size)
    }
    
    # Prepare Samples
    samples = get_s1_s3_samples(os.path.join(args.data_dir, 'test.txt'), stages, G, num_samples=args.num_samples)
    
    # Run All Analyses
    analyze_atomic_confidence(models, samples, stoi, args.device, args.output_dir)
    analyze_decomposition_paradox(models, samples, stoi, itos, stages, args.device, args.output_dir)
    analyze_geometry(models, samples, stoi, args.device, args.output_dir)
    analyze_weights(models, args.output_dir)
    
    print(f"\nFinal analysis complete! All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()