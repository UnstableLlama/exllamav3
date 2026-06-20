"""
Before/after demo for a QLoRA adapter trained on an EXL3 model.

Generates from the base model and from the adapted model for the same prompts,
so the effect of fine-tuning is obvious side by side. With the default pirate
dataset, the adapted model should answer in pirate speak.

Usage (CUDA box, after running examples/qlora_train.py):
    python examples/qlora_infer.py \
        --model   /path/to/exl3_model \
        --adapter out/exl3_qlora_adapter

NOTE: targets a real GPU + model; not executed in the authoring sandbox.
"""

import argparse
import torch


PROMPTS = [
    "Tell me about your morning.",
    "What is the best way to learn programming?",
    "Describe the weather today.",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(text, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(model.device)
    else:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    prev = model.config.use_cache
    model.config.use_cache = True
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.9, temperature=0.8,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    model.config.use_cache = prev
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from exllamav3.integration.transformers import patch_transformers
    from exllamav3.training import load_lora_adapter

    patch_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- BASE ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", dtype=torch.float16)
    model.eval()
    print("=" * 70)
    print("BASE MODEL")
    print("=" * 70)
    base_out = {p: generate(model, tokenizer, p, args.max_new_tokens) for p in PROMPTS}
    for p in PROMPTS:
        print(f"\n> {p}\n{base_out[p]}")

    # --- ADAPTED (attach trained adapter in place) ---
    load_lora_adapter(model, args.adapter, compute_dtype=torch.bfloat16)
    model.eval()
    print("\n" + "=" * 70)
    print("ADAPTED MODEL (QLoRA)")
    print("=" * 70)
    for p in PROMPTS:
        print(f"\n> {p}\n{generate(model, tokenizer, p, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
