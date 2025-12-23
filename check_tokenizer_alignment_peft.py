from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    BASE_MODEL = "Qwen/Qwen2.5-3B"
    CKPT_DIR = Path("out/qwen_composition_20251223_174604/ckpt_800")  # 改成你的
    tok = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True, trust_remote_code=True)
    print("tokenizer len =", len(tok))
    print("pad/eos:", tok.pad_token_id, tok.eos_token_id, tok.pad_token, tok.eos_token)

    # 如果是 adapter 目录
    if (CKPT_DIR / "adapter_config.json").exists():
        from peft import PeftModel

        cfg = json.loads((CKPT_DIR / "adapter_config.json").read_text())
        base_path = cfg.get("base_model_name_or_path", BASE_MODEL)
        print("adapter detected, base_model_name_or_path =", base_path)

        base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", trust_remote_code=True)

        # 关键：为了能 load 你这个 adapter（它保存了 embedding 权重 151666），这里必须 resize 到 tokenizer len
        # （后面我会教你怎么从根上避免“缩词表后遗症”）
        base.resize_token_embeddings(len(tok))

        model = PeftModel.from_pretrained(base, CKPT_DIR, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(CKPT_DIR, torch_dtype="auto", trust_remote_code=True)

    emb = model.get_input_embeddings().weight.shape[0]
    head = model.get_output_embeddings().weight.shape[0]
    print("model emb rows =", emb)
    print("lm_head rows  =", head)

if __name__ == "__main__":
    main()