import pickle
from pathlib import Path

DATA_DIR = Path("data/datasets/graph5hop_pg030_nps30_ns5_seed42_100P0")  # 改成你的
TRAIN_PATHS_PER_PAIR = 20

def load_pairs(path: Path):
    pairs=[]
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln=ln.strip()
        if not ln: 
            continue
        a=ln.split()
        if len(a)<2: 
            continue
        pairs.append((int(a[0]), int(a[1])))
    return pairs

def main():
    stage = pickle.load(open(DATA_DIR/"stage_info.pkl","rb"))
    stages = stage["stages"]
    nodes = [int(n) for st in stages for n in st]
    print("stage nodes min/max:", min(nodes), max(nodes), "count:", len(nodes))

    train_pairs = load_pairs(DATA_DIR/f"train_{TRAIN_PATHS_PER_PAIR}.txt")
    test_pairs  = load_pairs(DATA_DIR/"test.txt")
    all_nodes = [n for (s,t) in (train_pairs+test_pairs) for n in (s,t)]
    print("pair nodes min/max:", min(all_nodes), max(all_nodes))
    print("num_stages:", len(stages))

if __name__ == "__main__":
    main()