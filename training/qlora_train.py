"""
YAML-driven QLoRA training launcher.

Single command entry point for the native EXL3 QLoRA trainers. ``method:``
selects the objective -- ``sft`` (next-token CE, the default) or ``ebft``
(Energy-Based Fine-Tuning) -- and ``parallel:`` selects ``single``, ``split`` or
``ddp``; this launcher then execs the matching backend with the corresponding
command-line arguments. ``method: ebft`` runs the EBFT backend and supports
``parallel: single|split`` only (no ddp). For DDP, run this script directly (not
under torchrun): it will launch torchrun using the ``ddp`` section in the config.

Usage:
    python training/qlora_train.py --config training/qlora_train_config.yaml
    python training/qlora_train.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_BACKEND = SCRIPT_DIR / "qlora_train_native.py"
DDP_BACKEND = SCRIPT_DIR / "qlora_train_native_ddp.py"
EBFT_BACKEND = SCRIPT_DIR / "qlora_train_ebft.py"

METHOD_CHOICES = {"sft", "ebft"}


# Config keys that are launcher-only and must not be forwarded to backend scripts.
LAUNCHER_KEYS = {"ddp", "backend", "config", "parallel", "method"}
DDP_LAUNCH_KEYS = {
    "nproc_per_node", "nproc", "standalone", "nnodes", "node_rank",
    "master_addr", "master_port", "rdzv_backend", "rdzv_endpoint", "rdzv_id",
}

# Backend support differences. Common keys are forwarded to both, single-only keys
# are only valid for parallel=single|split, and ddp-only keys are only valid for
# parallel=ddp. Keep this explicit so a typo or unsupported DDP knob fails early
# instead of being silently ignored.
COMMON_KEYS = {
    "model", "out", "r", "alpha", "expert_r", "lora_dropout", "lr", "weight_decay", "scheduler",
    "warmup_ratio", "warmup_steps", "epochs", "steps", "batch", "grad_accum",
    "dataset", "dataset_split", "instruction_key", "context_key", "response_key",
    "messages_key", "prompt_format", "clean_text", "no_clean_text",
    "min_response_words", "uppercase_response", "max_samples", "shuffle",
    "shuffle_seed", "seq_len", "pack", "pack_algo", "ga_loss",
    "profile_dequant", "dequant_mode", "dequant_cache",
    "targets", "train_embeddings",
    "train_head", "compute_dtype", "no_grad_ckpt", "attn_impl", "ce_chunk",
    "head_vocab_chunk", "max_grad_norm", "save_every", "checkpoint_every",
    "keep_checkpoints", "resume", "reset_optimizer", "eval_split",
    "eval_dataset", "eval_config", "eval_text_key", "eval_max_samples",
    "eval_max_blocks",
    "eval2_dataset", "eval2_split", "eval2_config",
    "eval2_text_key", "eval2_max_samples", "eval2_max_blocks", "val_frac",
    "eval_every", "save_best", "run_log",
    # Ported to the DDP backend (Session 44): init computed on rank 0 and
    # broadcast; report/samples are rank-0-only there.
    "use_rslora", "init_lora", "init_svd_niter", "init_ref_model",
    "init_eva_tokens",
    "sample_every", "sample_prompt",
    "no_report", "run_name",
}
SINGLE_ONLY_KEYS = {
    "device", "parallel", "reserve_per_device", "use_per_device", "split_even", "optim",
    "inspect", "lora_embed", "lora_head", "offload_embed_head_optim",
    "offload_activations", "offload_mode", "vram_spillover", "use_liger",
    "quant_aware", "quant_aware_scale", "quant_aware_ref_model",
    "torch_profile",   # torch.profiler window; DDP backend has no such flag
    "wandb_project", "wandb_run_name", "wandb_entity",  # not mirrored to DDP yet
}
DDP_ONLY_KEYS = set()

# Keys forwarded to the EBFT backend (qlora_train_ebft.py) when method: ebft.
# Explicit so an SFT-only knob under method: ebft fails early instead of being
# silently dropped. ``parallel`` is appended by the launcher, not from here.
EBFT_KEYS = {
    # identity / placement
    "model", "out", "device", "reserve_per_device", "use_per_device",
    # LoRA / init
    "r", "alpha", "use_rslora", "targets", "expert_r", "init_lora",
    "init_svd_niter", "init_ref_model", "init_eva_tokens",
    # EBFT objective
    "gen_len", "n_samples", "anchors", "min_context", "temperature",
    "top_k", "top_p", "align_coef", "div_coef", "rl_coef", "ce_coef",
    "no_whiten", "whiten_tol", "feature_fracs",
    "rollout_sampler", "sampler_cache_tokens",
    # optimization
    "lr", "weight_decay", "optim", "adam_betas", "scheduler", "warmup_ratio",
    "warmup_steps", "epochs", "steps", "batch", "grad_accum", "max_grad_norm",
    # data
    "mode", "dataset", "dataset_split", "dataset_config", "instruction_key",
    "context_key", "response_key", "messages_key", "text_key", "prompt_format",
    "max_samples", "shuffle", "shuffle_seed", "seq_len", "clean_text",
    "min_response_words",
    # eval / saving
    "eval_split", "eval_dataset", "eval_max_samples", "val_frac", "eval_every",
    "no_eval_cfm", "eval_cfm_samples", "save_best", "save_every", "checkpoint_every",
    "keep_checkpoints", "resume", "reset_optimizer", "run_log", "seed",
    "self_test",
    # runtime knobs shared with the SFT trainer
    "compute_dtype", "no_grad_ckpt", "attn_impl", "ce_chunk", "head_vocab_chunk",
    "offload_activations", "offload_mode", "use_liger", "dequant_mode",
    # live sample generations (shared with the SFT trainer)
    "sample_every", "sample_prompt",
    # local run report
    "no_report", "run_name",
}

# EBFT-only knobs at their reference-code defaults. Mirrors SINGLE_ONLY_DEFAULTS:
# lets the fully-commented sample config expose the EBFT section at its defaults
# under method: sft (where these keys aren't forwarded) without tripping the
# unsupported-key check -- only a non-default EBFT value under sft is an error.
EBFT_DEFAULTS = {
    "mode": "qa", "gen_len": 8, "n_samples": 4, "anchors": 4, "min_context": 8,
    "temperature": 0.6, "top_k": 0, "top_p": 1.0,
    "align_coef": 1.0, "div_coef": 0.5, "rl_coef": 1.0, "ce_coef": 0.03,
    "no_whiten": False, "whiten_tol": 1e-5, "feature_fracs": "0.25,0.5,0.75",
    "adam_betas": "0.9,0.95", "no_eval_cfm": False, "eval_cfm_samples": 0,
    "rollout_sampler": "exact", "sampler_cache_tokens": 32768,
}
ALIASES = {"lora_r": "r"}  # DDP backend spells this --lora-r; config uses r.

SINGLE_ONLY_DEFAULTS = {
    "device": "cuda:0",
    "reserve_per_device": None,
    "use_per_device": None,
    "split_even": None,
    "optim": "adamw",
    "inspect": 0,
    "lora_embed": False,
    "lora_head": False,
    "offload_embed_head_optim": False,
    "offload_activations": False,
    "offload_mode": "async",
    "vram_spillover": False,
    "use_liger": False,
    "quant_aware": "none",
    "quant_aware_scale": 1.0,
    "quant_aware_ref_model": None,
    "wandb_project": "",
    "wandb_run_name": "",
    "wandb_entity": "",
}


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '\"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if ((value.startswith("'") and value.endswith("'")) or
            (value.startswith('\"') and value.endswith('\"'))):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for the flat config shape used by this launcher.

    Supports top-level ``key: value`` pairs plus one-level nested mappings (the
    ``ddp:`` section), booleans, nulls, numbers, strings and flow-style lists.
    If users need richer YAML, installing PyYAML enables that automatically.
    """
    root: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    current_indent = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            current_map = None
            current_indent = 0
            if ":" not in line:
                raise SystemExit(f"{path}:{lineno}: expected `key: value`")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                child: dict[str, Any] = {}
                root[key] = child
                current_map = child
                current_indent = None  # type: ignore[assignment]
            else:
                root[key] = _parse_scalar(value)
        else:
            if current_map is None:
                raise SystemExit(f"{path}:{lineno}: nested value without a parent mapping")
            if ":" not in line:
                raise SystemExit(f"{path}:{lineno}: expected nested `key: value`")
            key, value = line.strip().split(":", 1)
            current_map[key.strip()] = _parse_scalar(value.strip())
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        data = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a YAML mapping at the top level")
    return data


def normalize_key(key: str) -> str:
    key = key.replace("-", "_")
    return ALIASES.get(key, key)


def is_default_false(value: Any) -> bool:
    return value is False or value is None


def cli_flag(key: str, *, ddp: bool) -> str:
    if ddp and key == "r":
        return "--lora-r"
    return "--" + key.replace("_", "-")


def append_arg(argv: list[str], key: str, value: Any, *, ddp: bool) -> None:
    """Append one config key/value as backend CLI args."""
    if value is None:
        return
    flag = cli_flag(key, ddp=ddp)
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(v) for v in value)
        return
    argv.extend([flag, str(value)])


def flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize top-level config keys.

    The sample config is intentionally flat because it mirrors CLI flags one-to-one.
    ``ddp`` is the only nested section, reserved for torchrun launch options.
    """
    out: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = normalize_key(str(raw_key))
        if key in out:
            raise SystemExit(f"Duplicate config key after normalization: {raw_key!r}")
        out[key] = value
    return out


def validate_config(cfg: dict[str, Any], parallel: str, method: str) -> None:
    # Recognize the full schema in every mode so a full sample config can switch
    # between methods/parallel by changing only `method`/`parallel`; keys the
    # chosen backend does not accept are checked below and may remain present
    # when empty/false.
    supported = (COMMON_KEYS | SINGLE_ONLY_KEYS | DDP_ONLY_KEYS
                 | EBFT_KEYS | LAUNCHER_KEYS)
    unknown = sorted(k for k in cfg if k not in supported)
    if unknown:
        raise SystemExit("Unknown config key(s): " + ", ".join(unknown))

    if not cfg.get("model"):
        raise SystemExit("config must set `model: /path/to/exl3_model`")

    # The set of keys the chosen backend actually accepts.
    if method == "ebft":
        allowed, label = EBFT_KEYS, "method: ebft"
    elif parallel == "ddp":
        allowed, label = COMMON_KEYS | DDP_ONLY_KEYS, "parallel: ddp"
    else:
        allowed, label = COMMON_KEYS | SINGLE_ONLY_KEYS, "parallel: single|split"

    # A key the backend can't accept is an error only if it's set to a non-empty,
    # non-default value -- a full sample config may leave unsupported knobs at
    # their empty/false/default so it can switch backends by editing one key.
    bad = []
    for key in sorted(set(cfg) - allowed - LAUNCHER_KEYS):
        value = cfg[key]
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        if key in SINGLE_ONLY_DEFAULTS and value == SINGLE_ONLY_DEFAULTS[key]:
            continue
        if key in EBFT_DEFAULTS and value == EBFT_DEFAULTS[key]:
            continue
        if is_default_false(value):
            continue
        bad.append(key)
    if bad:
        raise SystemExit(
            f"These config key(s) are not supported by {label}: " + ", ".join(bad))

    if parallel != "split":
        for key in ("reserve_per_device", "use_per_device", "split_even"):
            if key in cfg and cfg[key]:
                raise SystemExit(f"{key} only applies when parallel: split")


def build_backend_argv(cfg: dict[str, Any], config_path: Path) -> list[str]:
    method = str(cfg.get("method", "sft")).lower()
    if method not in METHOD_CHOICES:
        raise SystemExit("method must be one of: " + ", ".join(sorted(METHOD_CHOICES)))
    parallel = str(cfg.get("parallel", "single")).lower()
    if parallel not in {"single", "split", "ddp"}:
        raise SystemExit("parallel must be one of: single, split, ddp")
    cfg["parallel"] = parallel
    if method == "ebft" and parallel == "ddp":
        raise SystemExit("method: ebft supports parallel: single|split only (no ddp).")
    validate_config(cfg, parallel, method)

    # EBFT: exec the EBFT backend, forwarding only the keys it accepts.
    if method == "ebft":
        argv: list[str] = [sys.executable, str(EBFT_BACKEND)]
        append_arg(argv, "parallel", parallel, ddp=False)  # accepts single|split
        for key in sorted(EBFT_KEYS):
            if key in cfg:
                append_arg(argv, key, cfg[key], ddp=False)
        return argv

    ddp = parallel == "ddp"
    backend = DDP_BACKEND if ddp else SINGLE_BACKEND
    argv = [sys.executable, str(backend)]

    if not ddp:
        # qlora_train_native.py accepts --parallel single|split.
        append_arg(argv, "parallel", parallel, ddp=False)

    forward_keys = COMMON_KEYS | DDP_ONLY_KEYS if ddp else COMMON_KEYS | SINGLE_ONLY_KEYS
    for key in sorted(forward_keys):
        if key not in cfg or key in LAUNCHER_KEYS or key == "parallel":
            continue
        append_arg(argv, key, cfg[key], ddp=ddp)

    if ddp:
        ddp_cfg = cfg.get("ddp") or {}
        if not isinstance(ddp_cfg, dict):
            raise SystemExit("`ddp` must be a mapping when provided")
        unknown_ddp = sorted(str(k) for k in ddp_cfg if str(k) not in DDP_LAUNCH_KEYS)
        if unknown_ddp:
            raise SystemExit("Unknown ddp launch key(s): " + ", ".join(unknown_ddp))
        nproc = ddp_cfg.get("nproc_per_node", ddp_cfg.get("nproc", 2))
        torchrun = ["torchrun"]
        if ddp_cfg.get("standalone", True):
            torchrun.append("--standalone")
        torchrun.extend(["--nproc_per_node", str(nproc)])
        for opt in ("nnodes", "node_rank", "master_addr", "master_port", "rdzv_backend", "rdzv_endpoint", "rdzv_id"):
            if opt in ddp_cfg and ddp_cfg[opt] is not None:
                torchrun.extend(["--" + opt.replace("_", "-"), str(ddp_cfg[opt])])
        argv = torchrun + argv[1:]

    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="YAML-driven QLoRA training launcher")
    ap.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Print the backend command and exit")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = flatten_config(load_yaml(config_path))
    argv = build_backend_argv(cfg, config_path)
    print("[qlora_train] " + " ".join(shlex.quote(a) for a in argv), flush=True)
    if args.dry_run:
        return
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
