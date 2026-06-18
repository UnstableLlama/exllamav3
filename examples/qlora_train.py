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
    ap.add_argument("--no-fused-ce", action="store_true",
                    help="Use the standard HF head/loss instead of fused cross-entropy")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable gradient checkpointing")
    ap.add_argument("--ce-chunk", type=int, default=1024,
                    help="Token chunk size for fused cross-entropy")
    args = ap.parse_args()

    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)
    from exllamav3.integration.transformers import patch_transformers
    from exllamav3.training import (attach_qlora, save_lora_adapter,
                                   prepare_model_for_qlora_training,
                                   qlora_causal_lm_loss)

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

    # 3. Memory: gradient checkpointing (+ input-grad hook, use_cache off).
    prepare_model_for_qlora_training(
        model, use_gradient_checkpointing=not args.no_grad_ckpt
    )

    # 4. Data + Trainer. With fused cross-entropy we override compute_loss so
    #    the [tokens x vocab] logits are never materialised.
    ds = build_tiny_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ce_chunk = args.ce_chunk
    use_fused_ce = not args.no_fused_ce

    class QLoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if not use_fused_ce:
                return super().compute_loss(model, inputs, return_outputs, **kwargs)
            labels = inputs["input_ids"] if "labels" not in inputs else inputs["labels"]
            loss = qlora_causal_lm_loss(
                model,
                input_ids=inputs["input_ids"],
                labels=labels,
                attention_mask=inputs.get("attention_mask"),
                chunk=ce_chunk,
            )
            return (loss, None) if return_outputs else loss

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
        gradient_checkpointing=False,  # already enabled via prepare_* above
        # Adam state covers only the few-MB adapter, so this stays cheap.
    )
    trainer = QLoRATrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator,
    )
    trainer.train()

    # 4. Save adapter in PEFT format (loadable for inference).
    save_lora_adapter(model, args.out, base_model_name_or_path=args.model)
    print(f"Done. Adapter written to {args.out}")


if __name__ == "__main__":
    main()
