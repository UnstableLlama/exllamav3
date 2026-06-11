#!/usr/bin/env python3
"""
Lightweight smoke checks for DiffusionGemma block-diffusion architecture integration.

Checks:
1) Config/architecture resolution and key prefix detection
2) Text model graph construction (self-conditioning module inserted, index fixups)
3) One sliding-attn block forward
4) One full-attn block forward (K==V, 512-dim heads)
5) Self-conditioning forward: encoder-mode passthrough, decoder-mode with and without logits

Optional:
6) Full model load and block-diffusion generation (--generate)
"""

import argparse
import sys
import torch

from exllamav3 import BlockDiffusionGenerator, BlockDiffusionSettings, Cache, Config, Model, Tokenizer


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.25,0.25")
    args = ap.parse_args()

    cfg = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture != "DiffusionGemmaForBlockDiffusion":
        fail(f"Unexpected architecture: {cfg.architecture}")
    print("[INFO] text_key_prefix:", cfg.text_key_prefix)
    print("[INFO] sc_key_prefix:", cfg.sc_key_prefix)
    print("[INFO] vision_key_prefix:", cfg.vision_key_prefix)
    print("[INFO] canvas_length:", cfg.canvas_length)
    if not cfg.enable_moe_block:
        fail("Expected enable_moe_block to be forced on")
    if not cfg.attention_k_eq_v:
        fail("Expected attention_k_eq_v to be forced on")

    model = Model.from_config(cfg)
    tokenizer = Tokenizer.from_config(cfg)
    if not model.caps.get("block_diffusion"):
        fail("Model missing block_diffusion capability")
    print("[INFO] model graph ok")

    sc_module = model.modules[1]
    if sc_module.module_name != "SelfConditioning":
        fail(f"Expected SelfConditioning at index 1, found {sc_module.module_name}")
    if model.first_block_idx != 2:
        fail(f"Expected first_block_idx == 2, got {model.first_block_idx}")
    blk0 = model.modules[model.first_block_idx]
    if not blk0.key.endswith(".layers.0"):
        fail(f"Expected first block key to end with .layers.0, got {blk0.key}")
    logits_module = model.modules[model.logit_layer_idx]
    if not logits_module.caps.get("logits_output"):
        fail("logit_layer_idx does not point at the logits layer")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size

    idx_sliding = model.first_block_idx + cfg.layer_types.index("sliding_attention")
    blk_sliding = model.modules[idx_sliding]
    blk_sliding.load(device)
    if blk_sliding.attn.v_proj is None:
        fail("Expected sliding attention to have a v_proj")
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_sliding.forward(x, params={})
    print("[INFO] sliding block forward:", blk_sliding.key, tuple(y.shape))
    blk_sliding.unload()

    idx_full = model.first_block_idx + cfg.layer_types.index("full_attention")
    blk_full = model.modules[idx_full]
    blk_full.load(device)
    if blk_full.attn.v_proj is not None:
        fail("Expected full attention to omit v_proj (K==V)")
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_full.forward(x, params={})
    print("[INFO] full block forward:", blk_full.key, tuple(y.shape))
    blk_full.unload()

    # Self-conditioning: encoder-mode passthrough, decoder-mode transforms
    embedding = model.modules[0]
    embedding.load(device)
    sc_module.load(device)
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.float)
    with torch.inference_mode():
        y_enc = sc_module.forward(x, params={})
        if not torch.equal(y_enc, x):
            fail("SelfConditioning is not a passthrough in encoder mode")
        y_first = sc_module.forward(x, params={"diffusion_decode": True, "self_conditioning_logits": None})
        if torch.equal(y_first, x):
            fail("SelfConditioning post-norm should transform the input in decoder mode")
        sc_logits = torch.randn((1, 8, cfg.vocab_size), device=device, dtype=torch.half)
        y_sc = sc_module.forward(x, params={"diffusion_decode": True, "self_conditioning_logits": sc_logits})
        if y_sc.shape != x.shape:
            fail(f"SelfConditioning output shape mismatch: {tuple(y_sc.shape)}")
        if torch.equal(y_sc, y_first):
            fail("Self-conditioning signal had no effect")
    print("[INFO] self-conditioning forward ok")
    sc_module.unload()
    embedding.unload()

    if args.generate:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(v) for v in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        model.load(
            reserve_per_device=reserve,
            progressbar=True,
            max_output_size=cfg.canvas_length,
        )
        cache = Cache(model, max_num_tokens=2048)
        generator = BlockDiffusionGenerator(model, cache, tokenizer)
        print("[INFO] settings:", generator.settings)
        result = generator.generate(
            prompt=model.default_chat_prompt("What is the capital of France? Answer with one word."),
            max_new_tokens=256,
            seed=1234,
        )
        print("[INFO] output:", repr(result["text"]))
        print(
            f"[INFO] {result['new_ids'].shape[-1]} tokens, {result['num_canvases']} canvas(es), "
            f"{result['denoising_steps']} steps, {result['tokens_per_forward']:.2f} tokens/forward"
        )
        if "paris" not in result["text"].lower():
            fail(f"Unexpected generation output: {result['text']!r}")
        model.unload()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] diffusion_gemma smoke checks complete")


if __name__ == "__main__":
    main()
