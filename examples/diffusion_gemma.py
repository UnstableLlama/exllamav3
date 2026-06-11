import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from exllamav3 import BlockDiffusionGenerator, BlockDiffusionSettings, Cache, Config, Model, Tokenizer
from exllamav3.util import Timer

"""
Block-diffusion text generation with DiffusionGemma.

The model denoises 256-token canvases in parallel rather than sampling one token at a time. With
--show_drafts, the evolving argmax canvas is printed after each denoising step, similar to the draft view
of HF's TextDiffusionStreamer.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", required = True, help = "Path to EXL3 (or unquantized) model")
    parser.add_argument("-p", "--prompt", default = "Why is the sky blue?")
    parser.add_argument("-n", "--max_new_tokens", type = int, default = 256)
    parser.add_argument("-c", "--cache_size", type = int, default = 4096)
    parser.add_argument("--steps", type = int, default = None, help = "Max denoising steps per canvas")
    parser.add_argument("--entropy_bound", type = float, default = None)
    parser.add_argument("--seed", type = int, default = None)
    parser.add_argument("--show_drafts", action = "store_true", help = "Print intermediate canvases")
    args = parser.parse_args()

    # Load model. Block diffusion models can also be used through the regular Generator/Job API; this
    # example uses the standalone loop for the per-step draft visualization
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)

    settings = BlockDiffusionSettings.from_directory(args.model_dir)
    if args.steps is not None:
        settings.max_denoising_steps = args.steps
    if args.entropy_bound is not None:
        settings.entropy_bound = args.entropy_bound

    generator = BlockDiffusionGenerator(model, cache, tokenizer, settings)
    prompt = model.default_chat_prompt(args.prompt)

    def on_draft(step, argmax_canvas):
        if not args.show_drafts:
            return
        text = tokenizer.decode(argmax_canvas.cpu(), decode_special_tokens = False)
        text = text[0] if isinstance(text, list) else text
        print(f"\x1b[2J\x1b[H-- draft, step {step + 1}:\n{text}", flush = True)

    def on_canvas(new_ids):
        if args.show_drafts:
            return
        text = tokenizer.decode(new_ids.cpu(), decode_special_tokens = False)
        text = text[0] if isinstance(text, list) else text
        print(text, end = "", flush = True)

    torch.cuda.synchronize()
    with Timer() as t:
        result = generator.generate(
            prompt = prompt,
            max_new_tokens = args.max_new_tokens,
            seed = args.seed,
            on_draft = on_draft,
            on_canvas = on_canvas,
        )
    torch.cuda.synchronize()

    if args.show_drafts:
        print(f"\x1b[2J\x1b[H-- final output:\n{result['text']}")
    print()
    print("---")
    num_tokens = result["new_ids"].shape[-1]
    print(
        f"{num_tokens} tokens, {result['num_canvases']} canvas(es), "
        f"{result['denoising_steps']} denoising steps "
        f"({result['tokens_per_forward']:.2f} tokens/forward), "
        f"{num_tokens / t.interval:.2f} tokens/second"
    )

if __name__ == "__main__":
    main()
