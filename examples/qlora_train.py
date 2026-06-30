"""
YAML-driven QLoRA training launcher.

Single command entry point for the native EXL3 QLoRA trainers. The YAML selects
``parallel: single``, ``parallel: split`` or ``parallel: ddp``; this launcher then
execs the matching backend with the corresponding command-line arguments. For
DDP, run this script directly (not under torchrun): it will launch torchrun using
the ``ddp`` section in the config.

Usage:
    python examples/qlora_train.py --config examples/qlora_train_config.yaml
    python examples/qlora_train.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SINGLE_BACKEND = ROOT / "examples" / "qlora_train_native.py"
DDP_BACKEND = ROOT / "examples" / "qlora_train_native_ddp.py"


# Config keys that are launcher-only and must not be forwarded to backend scripts.
LAUNCHER_KEYS = {"ddp", "backend", "config", "parallel"}
DDP_LAUNCH_KEYS = {
    "nproc_per_node", "nproc", "standalone", "nnodes", "node_rank",
    "master_addr", "master_port", "rdzv_backend", "rdzv_endpoint", "rdzv_id",
}

# Backend support differences. Common keys are forwarded to both, single-only keys
# are only valid for parallel=single|split, and ddp-only keys are only valid for
# parallel=ddp. Keep this explicit so a typo or unsupported DDP knob fails early
# instead of being silently ignored.
COMMON_KEYS = {
    "model", "out", "r", "alpha", "lr", "weight_decay", "scheduler",
    "warmup_ratio", "warmup_steps", "epochs", "steps", "batch", "grad_accum",
    "dataset", "dataset_split", "instruction_key", "context_key", "response_key",
    "messages_key", "prompt_format", "clean_text", "no_clean_text",
    "min_response_words", "uppercase_response", "max_samples", "shuffle",
    "shuffle_seed", "seq_len", "pack", "targets", "train_embeddings",
    "train_head", "compute_dtype", "no_grad_ckpt", "attn_impl", "ce_chunk",
    "head_vocab_chunk", "max_grad_norm", "save_every", "checkpoint_every",
    "keep_checkpoints", "resume", "reset_optimizer", "eval_split",
    "eval_dataset", "eval2_dataset", "eval2_split", "eval2_config",
    "eval2_text_key", "eval2_max_samples", "eval2_max_blocks", "val_frac",
    "eval_every", "save_best", "run_log",
}
SINGLE_ONLY_KEYS = {
    "device", "parallel", "reserve_per_device", "use_per_device", "optim",
    "inspect", "lora_embed", "lora_head", "offload_embed_head_optim",
    "offload_activations", "use_liger", "sample_every", "sample_prompt",
}
DDP_ONLY_KEYS = set()
ALIASES = {"lora_r": "r"}  # DDP backend spells this --lora-r; config uses r.

SINGLE_ONLY_DEFAULTS = {
    "device": "cuda:0",
    "reserve_per_device": None,
    "use_per_device": None,
    "optim": "adamw",
    "inspect": 0,
    "lora_embed": False,
    "lora_head": False,
    "offload_embed_head_optim": False,
    "offload_activations": False,
    "use_liger": False,
    "sample_every": 25,
    "sample_prompt": "Tell me about your day.",
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


def validate_config(cfg: dict[str, Any], parallel: str) -> None:
    # Recognize the full schema in every mode so a full sample config can switch
    # between single/split/ddp by changing only `parallel`; mode-specific keys are
    # checked below and may remain present when empty/false.
    supported = COMMON_KEYS | SINGLE_ONLY_KEYS | DDP_ONLY_KEYS | LAUNCHER_KEYS
    unknown = sorted(k for k in cfg if k not in supported)
    if unknown:
        raise SystemExit("Unknown config key(s): " + ", ".join(unknown))

    if parallel == "ddp":
        bad = []
        for key in sorted(SINGLE_ONLY_KEYS & set(cfg)):
            if key == "parallel":
                continue
            value = cfg[key]
            # Allow sample config to expose unsupported false/empty knobs without
            # making every DDP config delete them.
            if isinstance(value, (list, tuple, dict)) and not value:
                continue
            if key in SINGLE_ONLY_DEFAULTS and value == SINGLE_ONLY_DEFAULTS[key]:
                continue
            if not is_default_false(value):
                bad.append(key)
        if bad:
            raise SystemExit(
                "These config key(s) are only supported by parallel=single|split, "
                "not ddp: " + ", ".join(bad)
            )

    if parallel != "split":
        for key in ("reserve_per_device", "use_per_device"):
            if key in cfg and cfg[key]:
                raise SystemExit(f"{key} only applies when parallel: split")

    if not cfg.get("model"):
        raise SystemExit("config must set `model: /path/to/exl3_model`")


def build_backend_argv(cfg: dict[str, Any], config_path: Path) -> list[str]:
    parallel = str(cfg.get("parallel", "single")).lower()
    if parallel not in {"single", "split", "ddp"}:
        raise SystemExit("parallel must be one of: single, split, ddp")
    cfg["parallel"] = parallel
    validate_config(cfg, parallel)

    ddp = parallel == "ddp"
    backend = DDP_BACKEND if ddp else SINGLE_BACKEND
    argv: list[str] = [sys.executable, str(backend)]

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
