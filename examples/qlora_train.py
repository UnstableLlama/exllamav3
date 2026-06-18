"""
Minimal QLoRA fine-tuning of an EXL3 model via HuggingFace Trainer.

This trains low-rank adapters on top of a frozen EXL3-quantized model. Only
the linear layers are EXL3; every other component (norms, attention, RoPE,
the LM head loss) is stock Transformers, so the whole graph is differentiable
once the EXL3 linears expose a differentiable forward -- which
`exllamav3.training.attach_qlora` provides.

Requirements (on a CUDA box with the exllamav3 extension built):
    pip install transformers datasets accelerate

Usage:
    python examples/qlora_train.py \
        --model /path/to/exl3_model \
        --out   out/exl3_qlora_adapter

The resulting adapter is saved in PEFT format and can be loaded back for
inference with exllamav3.model.lora.LoRA (or PEFT).

NOTE: This script targets a real GPU + model and has not been executed in the
authoring sandbox (no CUDA there). The training *mechanics* it relies on are
covered by tests/test_qlora_train_loop.py and tests/test_qlora_grad.py.
"""

import argparse
import torch


def build_tiny_dataset(tokenizer, n_examples: int = 64, seq_len: int = 64):
    """A throwaway in-memory dataset so the loop is self-contained."""
    from datasets import Dataset

    texts = [
        "ExLlamaV3 runs EXL3-quantized language models efficiently on consumer GPUs.",
        "QLoRA freezes the quantized base weights and trains only low-rank adapters.",
        "The trellis weight is reconstructed on the fly and treated as a constant.",
        "Gradients flow to the adapter and to the input, never through the quantizer.",
    ]
    rows = [{"text": texts[i % len(texts)]} for i in range(n_examples)]
    ds = Dataset.from_list(rows)

    def tok(batch):
        out = tokenizer(
            batch["text"], truncation=True, max_length=seq_len,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tok, batched=True, remove_columns=["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target module leaf names (default: attn+mlp projections)")
    args = ap.parse_args()

    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)
    from exllamav3.integration.transformers import patch_transformers
    from exllamav3.training import attach_qlora, save_lora_adapter

    # 1. Register the EXL3 quantizer with Transformers and load the model.
    patch_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Attach trainable LoRA adapters; freeze everything else.
    attach_qlora(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        compute_dtype=torch.bfloat16,
    )
    model.train()
    model.config.use_cache = False  # required when training with grad checkpointing

    # 3. Data + Trainer (we reuse HF's loop, optimizer, accumulation, etc.).
    ds = build_tiny_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        learning_rate=args.lr,
        logging_steps=1,
        bf16=True,
        report_to=[],
        save_strategy="no",
        # Adam state covers only the few-MB adapter, so this stays cheap.
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator,
    )
    trainer.train()

    # 4. Save adapter in PEFT format (loadable for inference).
    save_lora_adapter(model, args.out, base_model_name_or_path=args.model)
    print(f"Done. Adapter written to {args.out}")


if __name__ == "__main__":
    main()
