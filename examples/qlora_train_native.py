"""
QLoRA fine-tuning of an EXL3 model with NO HuggingFace Transformers in the loop.

This trains low-rank adapters on a frozen EXL3 model using exllamav3's own
weights and a transformers-free differentiable forward
(:class:`exllamav3.training.native_llama.NativeLlamaQLoRA`). It exists because
the Transformers-based path couples training to a specific transformers version
(the EXL3 Llama-3.2 weights were calibrated against 4.45 and 5.x mis-handles the
llama3 RoPE); the native path reuses the exact RoPE/norms/scale that
exllamav3's correct inference forward uses, so it can't be broken upstream.

Requirements (CUDA box with the exllamav3 extension built):
    pip install datasets            # note: NO transformers / accelerate needed

Usage:
    python examples/qlora_train_native.py \
        --model /path/to/exl3_model \
        --out   out/exl3_qlora_adapter

Defaults fine-tune on a Shakespeare style-transfer set (Roudranil/
shakespearean-and-modern-english-conversational-dataset): a modern-English
line in, the original Early-Modern-English play line out. The style is strong
and consistent, so an assistant trained on it answers everyday questions in
florid Shakespearean -- the before/after is unmistakable at scale 1.0.

The data loader is dataset-agnostic: it reads instruction / context / response
columns whose names are configurable via --instruction-key / --context-key /
--response-key, so swapping in another instruction set (e.g. the older
TeeZee/dolly-15k-pirate-speech, which uses instruction/context/response) needs
no code change. Validate first with examples/qlora_validate_native.py, then
check the trained adapter with examples/qlora_infer_native.py -- both are also
transformers-free.

The adapter is saved in PEFT format, loadable by exllamav3.model.lora.LoRA
(and by PEFT).
"""

import argparse
import random
import re
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training.native_llama import NativeLlamaQLoRA


EOT = "<|eot_id|>"

# Stage directions / inline actions, e.g. "[as CAMBIO]", "[TRINCULO grabs ...]",
# "*stares at the ceiling*". Style datasets built from play scripts carry these,
# and the model happily learns to emit them, producing disjoint non-answers.
_STAGE_DIR = re.compile(r"\[[^\]]*\]|\*[^*]*\*")
_WHITESPACE = re.compile(r"\s+")


def clean_style_text(s):
    """Strip stage directions and collapse runaway whitespace/newlines."""
    s = _STAGE_DIR.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def build_sft_examples(model, tokenizer, dataset_name, max_samples, seq_len,
                       instruction_key="instruction", context_key="context",
                       response_key="response", split="train",
                       clean_text=True, min_response_words=3):
    """
    Load an instruction dataset and tokenize for completion-only SFT using the
    model's native Llama-3 chat template. Prompt tokens are masked with -100 so
    loss is computed only over the (styled) response.

    Columns are addressed by name (instruction_key / context_key / response_key)
    so the loader is not tied to the Dolly schema; context_key may be absent in
    the dataset (treated as empty).

    clean_text strips stage directions / inline actions and normalizes
    whitespace (helps play-script style sets like the Shakespeare default, whose
    raw rows otherwise teach the model to emit "[stage directions]"). Rows whose
    cleaned response has fewer than min_response_words tokens are dropped.

    Returns a list of dicts with python int lists: input_ids / labels.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=0).select(range(max_samples))

    examples = []
    for ex in ds:
        instr = (ex.get(instruction_key) or "").strip()
        ctx = (ex.get(context_key) or "").strip()
        resp = (ex.get(response_key) or "").strip()
        if clean_text:
            instr, ctx, resp = (clean_style_text(instr), clean_style_text(ctx),
                                clean_style_text(resp))
        if not resp or len(resp.split()) < min_response_words:
            continue
        user = instr if not ctx else f"{instr}\n\n{ctx}"

        # default_chat_prompt() already includes <|begin_of_text|> and ends with
        # the assistant header, so encode specials and don't add another BOS.
        # Tokenize the prompt and the response SEPARATELY and concatenate, so the
        # prompt/response boundary is exact -- masking by the prompt-string length
        # is vulnerable to tokenizer boundary merges that mis-align the mask.
        prompt_text = model.default_chat_prompt(user)
        prompt_ids = tokenizer.encode(
            prompt_text, add_bos=False, encode_special_tokens=True
        )[0].tolist()
        resp_ids = tokenizer.encode(
            resp + EOT, add_bos=False, encode_special_tokens=True
        )[0].tolist()

        input_ids = (prompt_ids + resp_ids)[:seq_len]
        labels = [-100] * len(prompt_ids) + list(resp_ids)
        labels = labels[:seq_len]
        if all(l == -100 for l in labels):
            continue  # response got truncated away; skip
        examples.append({"input_ids": input_ids, "labels": labels})

    return examples


def collate(batch, pad_id):
    """Right-pad a batch; pad input_ids with pad_id, labels with -100."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def sample(model, cache, tokenizer, generator, prompt, max_new_tokens=48):
    """Quick native generation for live progress feedback."""
    text = model.default_chat_prompt(prompt)
    resp = generator.generate(
        prompt=text, max_new_tokens=max_new_tokens,
        add_bos=False, completion_only=True,
    )
    return resp.strip().replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=1000,
                    help="Training steps. ~steps*batch examples seen; aim for >=1 "
                         "epoch (dataset_size/batch) to pick up a strong style.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--dataset",
        default="Roudranil/shakespearean-and-modern-english-conversational-dataset",
        help="HF dataset id. Default is a Shakespeare style-transfer set.",
    )
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="translated_dialog",
                    help="Column holding the prompt/instruction (Dolly: 'instruction')")
    ap.add_argument("--context-key", default="context",
                    help="Optional extra-context column; absent columns are ignored")
    ap.add_argument("--response-key", default="og_response",
                    help="Column holding the target response (Dolly: 'response')")
    ap.add_argument("--no-clean-text", action="store_true",
                    help="Disable stripping of [stage directions]/*actions* and "
                         "whitespace normalization (on by default; helps play-script "
                         "style sets, leave off for code/markdown datasets)")
    ap.add_argument("--min-response-words", type=int, default=3,
                    help="Drop rows whose cleaned response is shorter than this")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target module leaf names (default: attn+mlp projections)")
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=25,
                    help="Generate a sample completion every N steps (0 to disable)")
    ap.add_argument("--sample-prompt", default="Tell me about your day.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Checkpoint the adapter to --out every N steps (0 = only "
                         "at the end). The adapter is also saved on Ctrl-C.")
    args = ap.parse_args()

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Load native model + tokenizer (the forward that's correct on EXL3).
    config = Config.from_directory(args.model)
    model = Model.from_config(config)

    # The KV cache must be created BEFORE model.load() so each attention layer
    # allocates its cache during loading; otherwise generation asserts on a
    # missing k_cache. Only needed for the live samples.
    cache = None
    if args.sample_every:
        from exllamav3 import Cache
        cache = Cache(model, max_num_tokens=4096)

    model.load(device=args.device, progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 2. Build the differentiable QLoRA model (frozen base + trainable adapters).
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        compute_dtype=cdt, gradient_checkpointing=not args.no_grad_ckpt,
    )
    net.train()
    print(f" -- trainable LoRA params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules})")

    # 3. Data.
    examples = build_sft_examples(
        model, tokenizer, args.dataset, args.max_samples, args.seq_len,
        instruction_key=args.instruction_key, context_key=args.context_key,
        response_key=args.response_key, split=args.dataset_split,
        clean_text=not args.no_clean_text,
        min_response_words=args.min_response_words,
    )
    print(f" -- {len(examples)} SFT examples")
    assert examples, "no usable training examples"

    # 4. Optional generator for live samples (KV-cache inference path). The cache
    #    was allocated before load() above.
    generator = None
    if args.sample_every:
        from exllamav3 import Generator
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
        net.eval()
        with torch.inference_mode():
            base = sample(model, cache, tokenizer, generator, args.sample_prompt)
        net.train()
        print(f"\n\U0001f3ad  baseline (step 0): {args.sample_prompt}\n     -> {base}\n")

    # 5. Optimizer over the adapter params only.
    opt = torch.optim.AdamW(net.lora_parameters(), lr=args.lr)

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(0).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    def adapter_b_norm():
        with torch.no_grad():
            return sum(w.lora_b.float().pow(2).sum() for w in net._wrappers
                       if w.r > 0).sqrt().item()

    def save(tag):
        # Always leave net in train mode after; saving touches the adapter only.
        net.save_adapter(args.out, base_model_name_or_path=args.model)
        print(f"{tag} Adapter written to {args.out}")

    bgen = batches()
    opt.zero_grad(set_to_none=True)
    ema = None
    step = 0
    try:
        for step in range(1, args.steps + 1):
            accum_loss = 0.0
            for _ in range(args.grad_accum):
                batch = next(bgen)
                input_ids, labels, attn = collate(batch, pad_id)
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk)
                (loss / args.grad_accum).backward()
                accum_loss += loss.item() / args.grad_accum

            # grad norm BEFORE clipping is a direct check that gradients reach the
            # adapters (a flat ~0 here would mean the backward graph is broken).
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.lora_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            opt.zero_grad(set_to_none=True)

            ema = accum_loss if ema is None else 0.9 * ema + 0.1 * accum_loss
            print(f"  step {step:>5}/{args.steps} | loss {accum_loss:6.4f} | "
                  f"ema {ema:6.4f} | grad {gnorm:7.4f} | |B| {adapter_b_norm():7.3f}")

            if args.sample_every and step % args.sample_every == 0:
                net.eval()
                net.apply_to_native()      # make generation reflect the adapter
                with torch.inference_mode():
                    txt = sample(model, cache, tokenizer, generator, args.sample_prompt)
                net.remove_from_native()
                net.train()
                print(f"\n  \U0001f3ad  [step {step}] {args.sample_prompt}\n     -> {txt}\n")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")
    except KeyboardInterrupt:
        # Stopping early at the loss plateau is a normal workflow; don't discard
        # the adapter trained so far.
        print(f"\nInterrupted at step {step}; saving adapter before exit.")
        if step > 0:
            save("[interrupted]")
        raise SystemExit(0)

    # 6. Save adapter (PEFT format; loadable by exllamav3.model.lora.LoRA).
    save("Done.")
    print("Verify with: python examples/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}")


if __name__ == "__main__":
    main()
