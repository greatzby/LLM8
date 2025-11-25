#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphA Mechanism Analysis Script
对比 SFT, PG, Q-Learning 模型的内部机制。

Usage:
    python analyze_mechanism.py \
        --data_dir data/datasets/graphA_pg030_tier3_P13_0 \
        --ckpt_sft out/composition_20251022_124024/ckpt_50000.pt \
        --ckpt_pg out_pg/pg_20251117_020015/ckpt_pg_18000.pt \
        --ckpt_ql out_ql_no_target/ql_20251119_151554/ckpt_ql_20000.pt \
        --output_dir analysis_results
"""

import argparse
import os
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from pathlib import Path
from model import GPT, GPTConfig

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 防止中文乱码兼容性问题，用通用字体

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_sft", type=str, required=True)
    parser.add_argument("--ckpt_pg", type=str, required=True)
    parser.add_argument("--ckpt_ql", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=500, help="用于分析的测试样本数量")
    return parser.parse_args()

def load_model(ckpt_path, device, meta_vocab_size):
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    
    # 确保 vocab_size 正确 (有时候 ckpt 里的 vocab_size 可能和 meta 不一致，以 meta 为准或保持一致)
    model_args['vocab_size'] = meta_vocab_size
    # 强制关闭 dropout 以获得确定性输出
    model_args['dropout'] = 0.0
    
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # 处理 state_dict (去除编译前缀等)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # 处理 block_size 可能不匹配的问题 (简单的截断或填充逻辑，参考你的训练脚本)
    # 这里简化处理：假设 ckpt 和 config 一致，或者直接加载
    # 如果报错，请参考训练脚本中的 load_state_dict_with_block_resize 函数
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Strict load failed, trying non-strict. Error: {e}")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model

def get_s1_s3_samples(test_file, stages, graph, num_samples=500):
    """从测试集中提取合法的 S1->S3 样本，并找到一个合法的中间点 S2 用于测试"""
    S1, S2, S3 = stages[0], stages[1], stages[2]
    samples = []
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
        
    # 随机打乱以获得多样性
    import random
    random.seed(42)
    random.shuffle(lines)
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue
        src, tgt = int(parts[0]), int(parts[1])
        
        if src in S1 and tgt in S3:
            # 我们需要找到一个合法的 S1->S2->S3 路径的前缀
            # 为了分析，我们手动构造一个正确的 S1->S2 步骤
            # 查找所有连接 src 和 tgt 的 s2 节点
            valid_s2 = []
            for s2_node in S2:
                if graph.has_edge(str(src), str(s2_node)) and graph.has_edge(str(s2_node), str(tgt)):
                    valid_s2.append(s2_node)
            
            if valid_s2:
                # 选第一个合法的 S2 节点作为 context
                s2_step = valid_s2[0]
                samples.append({
                    'src': src,
                    'tgt': tgt,
                    's2_step': s2_step
                })
                
        if len(samples) >= num_samples:
            break
    
    print(f"Collected {len(samples)} valid S1->S3 test samples.")
    return samples

# --- Analysis 1: Logits & Probability (The "Stop" Problem) ---
def analyze_logits(models, samples, stoi, itos, device, output_dir):
    print("Running Logits Analysis...")
    results = defaultdict(list)
    stop_token_id = stoi['\n']
    
    for sample in samples:
        # Context: [S1, S3, S1, S2_node]
        # 我们想看模型在生成 S2_node 之后，预测下一个 token 的概率分布
        # SFT 可能会预测 EOS (停止)
        # RL 应该预测 S3_node (继续)
        
        context = [
            stoi[str(sample['src'])],
            stoi[str(sample['tgt'])],
            stoi[str(sample['src'])],
            stoi[str(sample['s2_step'])]
        ]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        
        target_s3_token = stoi[str(sample['tgt'])]
        
        for model_name, model in models.items():
            with torch.no_grad():
                logits, _ = model(x)
            
            # 取最后一个 token 的 logits
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            p_eos = probs[stop_token_id].item()
            p_target = probs[target_s3_token].item()
            
            # 计算熵 (Entropy) - 衡量不确定性
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            results[model_name].append({
                'p_eos': p_eos,
                'p_target': p_target,
                'entropy': entropy
            })
            
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: P(EOS)
    data_eos = []
    for name, res in results.items():
        for r in res:
            data_eos.append({'Model': name, 'Probability': r['p_eos']})
    sns.boxplot(data=data_eos, x='Model', y='Probability', ax=axes[0], palette="Set2")
    axes[0].set_title('Probability of Stop Token (<EOS>) at S2')
    axes[0].set_ylabel('P(EOS)')

    # Plot 2: P(Target S3)
    data_tgt = []
    for name, res in results.items():
        for r in res:
            data_tgt.append({'Model': name, 'Probability': r['p_target']})
    sns.boxplot(data=data_tgt, x='Model', y='Probability', ax=axes[1], palette="Set2")
    axes[1].set_title('Probability of Correct Target S3 at S2')
    
    # Plot 3: Entropy
    data_ent = []
    for name, res in results.items():
        for r in res:
            data_ent.append({'Model': name, 'Entropy': r['entropy']})
    sns.boxplot(data=data_ent, x='Model', y='Entropy', ax=axes[2], palette="Set2")
    axes[2].set_title('Prediction Entropy at S2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logits_analysis.png'))
    plt.close()
    print("Logits analysis saved.")

# --- Analysis 2: Hidden States (Representation) ---
def get_last_hidden_state(model, x):
    # Hook function to capture output of the last block
    hidden_states = []
    def hook_fn(module, input, output):
        hidden_states.append(output)
    
    # Register hook on the last transformer block
    # model.transformer.h is a ModuleList
    handle = model.transformer.h[-1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(x)
        
    handle.remove()
    # output shape: [batch, seq_len, embed_dim]
    # We want the last token's embedding: [1, -1, embed_dim]
    return hidden_states[0][0, -1, :].cpu().numpy()

def analyze_hidden_states(models, samples, stoi, device, output_dir):
    print("Running Hidden State Analysis...")
    
    # 准备数据
    X_data = defaultdict(list)
    y_labels = [] # 这里的 label 是 Target S3 的 ID，用于看聚类
    
    # 为了可视化清晰，我们只选前 5-10 个最常见的 S3 目标进行可视化，否则颜色太多
    from collections import Counter
    tgt_counts = Counter([s['tgt'] for s in samples])
    top_targets = set([t for t, c in tgt_counts.most_common(8)])
    
    filtered_samples = [s for s in samples if s['tgt'] in top_targets]
    print(f"Filtered to {len(filtered_samples)} samples covering top 8 targets for visualization.")
    
    for sample in filtered_samples:
        context = [
            stoi[str(sample['src'])],
            stoi[str(sample['tgt'])],
            stoi[str(sample['src'])],
            stoi[str(sample['s2_step'])]
        ]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        
        for name, model in models.items():
            vec = get_last_hidden_state(model, x)
            X_data[name].append(vec)
            
        y_labels.append(str(sample['tgt'])) # Label is the target node ID
        
    # t-SNE Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (name, vectors) in enumerate(X_data.items()):
        vectors = np.array(vectors)
        # 先用 PCA 降维到 50 (如果维度很高)，这里维度只有 92/120，可以直接 t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(vectors)
        
        # Scatter plot
        sns.scatterplot(
            x=X_embedded[:,0], y=X_embedded[:,1], 
            hue=y_labels, 
            palette="tab10", 
            ax=axes[idx],
            legend=False, # 关闭图例防止遮挡，颜色代表不同的 S3 目标
            s=60, alpha=0.8
        )
        axes[idx].set_title(f"{name} Hidden Space (Colored by Target S3)")
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        
    # Add a single legend manually if needed, or just explain in caption
    plt.suptitle("t-SNE of Hidden States at S2 Node (Do they cluster by Goal?)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hidden_state_tsne.png'))
    plt.close()
    
    # Cosine Similarity Analysis (SFT vs RL)
    # 计算 SFT 和 RL 对应样本的余弦相似度
    print("Calculating Cosine Similarity between SFT and RL models...")
    sim_pg = []
    sim_ql = []
    
    vecs_sft = np.array(X_data['SFT'])
    vecs_pg = np.array(X_data['PG'])
    vecs_ql = np.array(X_data['QL'])
    
    def cosine_sim(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm1 * norm2 + 1e-9)
    
    for i in range(len(vecs_sft)):
        sim_pg.append(cosine_sim(vecs_sft[i], vecs_pg[i]))
        sim_ql.append(cosine_sim(vecs_sft[i], vecs_ql[i]))
        
    plt.figure(figsize=(8, 5))
    sns.kdeplot(sim_pg, label='SFT vs PG', fill=True)
    sns.kdeplot(sim_ql, label='SFT vs QL', fill=True)
    plt.title("Distribution of Cosine Similarity (Representation Drift)")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'representation_drift.png'))
    plt.close()
    print("Hidden state analysis saved.")

# --- Analysis 3: Weight Changes (Mechanism) ---
def analyze_weights(models, output_dir):
    print("Running Weight Analysis...")
    
    base_model = models['SFT']
    diff_results = []
    
    for name in ['PG', 'QL']:
        target_model = models[name]
        
        # 遍历每一层参数
        for (k1, v1), (k2, v2) in zip(base_model.named_parameters(), target_model.named_parameters()):
            assert k1 == k2, "Model architectures do not match!"
            
            # 计算 L2 距离 (Norm of difference)
            diff = (v1 - v2).norm().item()
            # 计算相对变化 (Relative change)
            rel_diff = diff / (v1.norm().item() + 1e-9)
            
            # Categorize layers
            layer_type = "Other"
            if "wte" in k1 or "wpe" in k1: layer_type = "Embedding"
            elif "attn" in k1: layer_type = "Attention"
            elif "mlp" in k1: layer_type = "MLP"
            elif "ln" in k1: layer_type = "LayerNorm"
            elif "lm_head" in k1: layer_type = "LM Head"
            
            # Extract layer number if exists
            layer_num = -1
            if ".h." in k1:
                try:
                    parts = k1.split('.')
                    h_idx = parts.index('h')
                    layer_num = int(parts[h_idx+1])
                except:
                    pass
            
            diff_results.append({
                'Method': name,
                'Layer Name': k1,
                'Layer Type': layer_type,
                'Layer Num': layer_num,
                'L2 Diff': diff,
                'Relative Diff': rel_diff
            })
            
    # Visualization
    import pandas as pd
    df = pd.DataFrame(diff_results)
    
    # 1. Overall Change Magnitude by Layer Type
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Layer Type', y='Relative Diff', hue='Method', errorbar=None)
    plt.title("Relative Weight Change by Module Type")
    plt.ylabel("Relative L2 Norm Difference")
    plt.savefig(os.path.join(output_dir, 'weight_change_by_type.png'))
    plt.close()
    
    # 2. Layer-wise Change (if multi-layer)
    # 如果只有1层，这个图可能没意义，但代码通用
    if df['Layer Num'].max() > 0:
        plt.figure(figsize=(12, 6))
        # Filter only transformer blocks
        df_blocks = df[df['Layer Num'] >= 0]
        sns.lineplot(data=df_blocks, x='Layer Num', y='Relative Diff', hue='Method', marker='o')
        plt.title("Weight Change across Layers")
        plt.savefig(os.path.join(output_dir, 'weight_change_layers.png'))
        plt.close()
        
    # 3. Top 10 Changed Parameters
    print("\nTop 5 Changed Parameters (Relative) for QL:")
    top_ql = df[df['Method'] == 'QL'].sort_values('Relative Diff', ascending=False).head(5)
    print(top_ql[['Layer Name', 'Relative Diff']])
    
    print("Weight analysis saved.")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Meta Info
    meta_path = os.path.join(args.data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    vocab_size = meta['vocab_size']
    
    stage_path = os.path.join(args.data_dir, 'stage_info.pkl')
    with open(stage_path, 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    graph_path = os.path.join(args.data_dir, 'composition_graph.graphml')
    G = nx.read_graphml(graph_path)
    
    # 2. Load Models
    models = {}
    models['SFT'] = load_model(args.ckpt_sft, args.device, vocab_size)
    models['PG'] = load_model(args.ckpt_pg, args.device, vocab_size)
    models['QL'] = load_model(args.ckpt_ql, args.device, vocab_size)
    
    # 3. Prepare Data
    test_file = os.path.join(args.data_dir, 'test.txt')
    samples = get_s1_s3_samples(test_file, stages, G, num_samples=args.num_samples)
    
    # 4. Run Analyses
    analyze_logits(models, samples, stoi, itos, args.device, args.output_dir)
    analyze_hidden_states(models, samples, stoi, args.device, args.output_dir)
    analyze_weights(models, args.output_dir)
    
    print(f"\nAll analyses complete! Check {args.output_dir} for images.")

if __name__ == "__main__":
    main()