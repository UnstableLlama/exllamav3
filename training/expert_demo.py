"""
MoE expert-adapter inference demo: BASE -> ADAPTED -> UNLOADED.

Loads a trained adapter (including routed-expert `expert_*_proj` tensors,
which take the unfused per-expert path at inference — Session 26) on the
native generation path, using the trainer's own chat formats so the adapter
sees the exact template it was trained on. Greedy sampling so the UNLOADED
generation can be compared byte-for-byte against BASE.

Caveat on the byte-compare: MoE routers pick non-deterministically on
near-tie logits, so on some models (observed on Qwen3.6-35B-A3B) even two
back-to-back BASE generations differ in a few words. A subtle UNLOADED-vs-BASE
diff on an MoE model is routing-tie noise, not an unload leak — rerun the
base twice to confirm before suspecting the loader.

Usage:
    python training/expert_demo.py \
        --model /path/to/exl3-moe-model --adapter out/my_expert_adapter \
        --prompt-format gemma4-nothink [--use-per-device 18 23]
"""

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.model.lora import LoRA
from exllamav3.generator.sampler import ComboSampler
from qlora_train_native import format_prompt_and_eot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt-format", required=True,
                    help="trainer chat format (e.g. auto, gemma4-nothink, "
                         "qwen3.5-nothink) -- must match training")
    ap.add_argument("--prompt", default="what is truth")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--use-per-device", type=int, nargs="+", default=None,
                    help="per-GPU GB budgets for a layer-split load "
                         "(omit = single device cuda:0)")
    ap.add_argument("--lora-scaling", type=float, default=1.0)
    args = ap.parse_args()

    sampler = ComboSampler(temperature=0.0)   # greedy: unload should match base

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    if args.use_per_device is not None:
        model.load(progressbar=True, use_per_device=args.use_per_device)
    else:
        model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    build_prompt, eot = format_prompt_and_eot(model, tokenizer, args.prompt_format)
    prompt = build_prompt(args.prompt)

    stop = list(getattr(config, "eos_token_id_list", None) or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop:
        stop.append(tokenizer.eos_token_id)
    stop.append(eot)

    def gen():
        return generator.generate(
            prompt=prompt, max_new_tokens=args.max_new_tokens, sampler=sampler,
            add_bos=False, completion_only=True, stop_conditions=stop)

    def show(label, text):
        print("=" * 70 + f"\n{label}\n" + "=" * 70 + f"\n{text}\n")

    base = gen()
    show("BASE", base)

    lora = LoRA.from_directory(model, args.adapter, lora_scaling=args.lora_scaling)
    adapted = gen()
    show("ADAPTED", adapted)

    lora.unload()
    restored = gen()
    show("UNLOADED", restored)

    print(f"adapted != base:      {adapted != base}")
    print(f"unloaded == base:     {restored == base}"
          + ("" if restored == base else "   (see MoE routing-tie caveat in the docstring)"))


if __name__ == "__main__":
    main()
