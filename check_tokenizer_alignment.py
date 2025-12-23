from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen2.5-3B"  # 你实际用的 base/ref 模型名
CKPT_DIR = "out/qwen_composition_20251223_174604/ckpt_800"  # 你日志里的 ckpt
DATA_TOK_DIR = "data/datasets/graph5hop_pg030_nps30_ns5_seed42_100P0/tokenizer"  # 你 prepare 保存的 tokenizer 目录

def show(tok, name):
    print("="*80)
    print(name)
    print("len(tok) =", len(tok))
    print("pad:", tok.pad_token, tok.pad_token_id)
    print("eos:", tok.eos_token, tok.eos_token_id)
    print("bos:", tok.bos_token, tok.bos_token_id)
    print("unk:", tok.unk_token, tok.unk_token_id)
    print("special_tokens_map:", tok.special_tokens_map)

def main():
    tok_base = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    tok_ckpt = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True, trust_remote_code=True)
    tok_data = AutoTokenizer.from_pretrained(DATA_TOK_DIR, use_fast=True, trust_remote_code=True)

    show(tok_base, "TOKENIZER: base model")
    show(tok_ckpt, "TOKENIZER: ckpt dir")
    show(tok_data, "TOKENIZER: data_dir/tokenizer")

    model = AutoModelForCausalLM.from_pretrained(CKPT_DIR, torch_dtype="auto", trust_remote_code=True)
    emb = model.get_input_embeddings().weight.shape[0]
    print("="*80)
    print("MODEL (ckpt) embedding rows =", emb)

    print("\nQuick verdict:")
    print("len(tok_base) == emb ?", len(tok_base) == emb)
    print("len(tok_ckpt) == emb ?", len(tok_ckpt) == emb)
    print("len(tok_data) == emb ?", len(tok_data) == emb)

if __name__ == "__main__":
    main()