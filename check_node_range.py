import json
from pathlib import Path

DATA_DIR = Path("data/datasets/graph5hop_pg030_nps30_ns5_seed42_100P0")  # 改成你的 data_dir
K = 5  # 你训练的 K

def load_pairs_unique(path: Path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: 
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        s = int(parts[0]); t = int(parts[1])
        pairs.append((s,t))
    pairs = sorted(set(pairs))
    return pairs

def main():
    stage_info_path = DATA_DIR / f"stage_info.pkl"
    train_path = DATA_DIR / f"train_{K}.txt"
    test_path = DATA_DIR / "test.txt"

    stage_info = json.loads(stage_info_path.read_text(encoding="utf-8"))
    node_to_stage = {int(n): int(st) for n, st in stage_info["node_to_stage"].items()}

    train_pairs = load_pairs_unique(train_path)
    test_pairs = load_pairs_unique(test_path)

    all_pair_nodes = [n for (s,t) in (train_pairs + test_pairs) for n in (s,t)]

    print("[pairs] count:", len(train_pairs), len(test_pairs))
    print("[pairs] min/max node:", min(all_pair_nodes), max(all_pair_nodes))
    print("[stage_info] min/max node:", min(node_to_stage.keys()), max(node_to_stage.keys()))
    print("[stage_info] stages min/max:", min(node_to_stage.values()), max(node_to_stage.values()))

if __name__ == "__main__":
    main()