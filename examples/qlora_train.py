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

By default this fine-tunes on TeeZee/dolly-15k-pirate-speech -- an instruction
dataset whose responses are rewritten in pirate speech ("Arrr, batten down the
hatches!"). It's a deliberately obvious style so the before/after is
unmistakable; see examples/qlora_infer.py to compare base vs adapted output.

The resulting adapter is saved in PEFT format and can be loaded back for
inference with exllamav3.model.lora.LoRA (or PEFT).

NOTE: This script targets a real GPU + model and has not been executed in the
authoring sandbox (no CUDA there). The training *mechanics* it relies on are
covered by tests/test_qlora_train_loop.py and tests/test_qlora_grad.py.
"""

import argparse
import torch


def build_sft_dataset(tokenizer, dataset_name: str, max_samples: int, seq_len: int):
    """
    Load an instruction/response dataset (Dolly schema) and tokenize it for
    completion-only SFT: the prompt tokens are masked with -100 so the loss is
    computed only over the (pirate) response.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=0).select(range(max_samples))

    has_chat = getattr(tokenizer, "chat_template", None) is not None

    def format_example(ex):
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        resp = (ex.get("response") or "").strip()
        user = instr if not ctx else f"{instr}\n\n{ctx}"

        if has_chat:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": resp}],
                tokenize=True, add_generation_prompt=False,
            )
        else:
            prompt = f"### Instruction:\n{user}\n\n### Response:\n"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(
                prompt + resp + (tokenizer.eos_token or ""),
                add_special_tokens=True,
            )["input_ids"]

        full_ids = full_ids[:seq_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(full_ids))):
            labels[i] = -100  # mask the prompt; train only on the response
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    return ds.map(format_example, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dataset", default="TeeZee/dolly-15k-pirate-speech",
                    help="HF instruction dataset (Dolly schema: instruction/context/response)")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=512)
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
                              DataCollatorForSeq2Seq)
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
    ds = build_sft_dataset(tokenizer, args.dataset, args.max_samples, args.seq_len)
    # DataCollatorForSeq2Seq pads input_ids with pad_token and labels with -100.
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

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
