"""
Native (no-transformers) before/after demo for a QLoRA adapter on EXL3.

The HF Transformers integration is only needed for *training* (autograd).
For inference we use exllamav3's own forward -- which works correctly on the
quantized weights -- and its native PEFT LoRA loader, sidestepping the HF
integration and any transformers version issues entirely.

Usage:
    python examples/qlora_infer_native.py \
        --model   /path/to/exl3_model \
        --adapter out/exl3_qlora_adapter
"""

import argparse
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.model.lora import LoRA
from exllamav3.generator.sampler import ComboSampler


# General everyday prompts: a style adapter is most convincing on plain,
# open-ended questions, where the learned voice (e.g. florid Shakespearean
# English) colours an ordinary answer rather than a meta/refusal one.
PROMPTS = [
    "Tell me about your day.",
    "Give me some advice about love.",
    "Explain how the water cycle works.",
    "What should I have for dinner tonight?",
]


def llama3_prompt(user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--lora-scaling", type=float, default=1.0,
                    help="Extra multiplier on the adapter (on top of alpha/r). "
                         ">1 amplifies the learned style to make a subtle adapter visible.")
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="Custom prompts (default: a content-rich built-in set)")
    # Sampling. The library default is temp 0.8 + min_p 0.08, which truncates the
    # low-probability tail -- if a style's markers are sparse/rare, they get cut.
    # Raise --temperature and set --min-p 0 to test whether the style is in the
    # tail (vs not learned). --temperature 0 = greedy/argmax.
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min-p", type=float, default=0.08)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed for reproducible before/after comparison")
    args = ap.parse_args()

    prompts = args.prompts or PROMPTS
    sampler = ComboSampler(
        temperature=args.temperature, min_p=args.min_p,
        top_p=args.top_p, top_k=args.top_k,
    )

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    def run(label: str):
        print("=" * 70)
        print(label)
        print("=" * 70)
        for p in prompts:
            resp = generator.generate(
                prompt=llama3_prompt(p),
                max_new_tokens=args.max_new_tokens,
                sampler=sampler,
                seed=args.seed,
                add_bos=False,
                completion_only=True,
            )
            print(f"\n> {p}\n{resp}")
        print()

    run("BASE MODEL (native exllamav3)")

    lora = LoRA.from_directory(model, args.adapter, lora_scaling=args.lora_scaling)
    run("ADAPTED MODEL (native exllamav3 + QLoRA)")


if __name__ == "__main__":
    main()
