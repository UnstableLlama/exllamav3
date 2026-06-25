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

Defaults fine-tune on superdrew100/UwU_Alpaca_data: the Alpaca-cleaned
instruction set with every answer rewritten in over-the-top "UwU" furry speak
(caps, emoji, "OwO", "*twitches whiskers*"). Because it keeps Alpaca's clean
question->on-topic-answer structure, the model stays coherent while the style
is unmistakable at scale 1.0 -- unlike play-script style sets, whose responses
are tangential monologues that teach the model to ramble. (Note: the persona
has mild PG-13 innuendo in places.)

The data loader is dataset-agnostic: it reads instruction / context / response
columns whose names are configurable via --instruction-key / --context-key /
--response-key, so swapping in another instruction set (e.g. Dolly-schema
TeeZee/dolly-15k-pirate-speech via --instruction-key instruction --context-key
context --response-key response) needs no code change. Validate first with
examples/qlora_validate_native.py, then check the trained adapter with
examples/qlora_infer_native.py -- both are also transformers-free.

The adapter is saved in PEFT format, loadable by exllamav3.model.lora.LoRA
(and by PEFT).
"""

import argparse
import csv
import datetime
import math
import os
import random
import re
import time
from collections import deque
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training.native_llama import NativeLlamaQLoRA


class ThroughputMeter:
    """Rolling tok/s over a sliding window of recent steps, for a live readout.

    Tracks supervised (loss-bearing, labels != -100) and total (non-pad) tokens
    separately, so the per-step line can show real throughput rather than the
    run-average the final ``[PERF]`` line reports. Window-based (not cumulative)
    so the number reflects current steady state, not warmup. Time fed in should
    be the train-step compute only (exclude eval/sample/save) for a clean rate.
    """

    def __init__(self, window=20):
        self.buf = deque(maxlen=window)   # (dt, supervised_tokens, total_tokens)

    def update(self, dt, supervised, total):
        self.buf.append((float(dt), int(supervised), int(total)))

    def rates(self):
        """Return (supervised_tok_per_s, total_tok_per_s) over the window."""
        tt = sum(b[0] for b in self.buf)
        if tt <= 0:
            return 0.0, 0.0
        return sum(b[1] for b in self.buf) / tt, sum(b[2] for b in self.buf) / tt


# Canonical schema for the per-run CSV log. Fixed order so the "mega CSV" stays
# consistent across runs/arms; the BNB arm inlines an identical copy (separate
# venv) and the DDP script imports these. Unknown keys are ignored and missing
# fields written blank, so adding a column later only needs an entry here.
RUN_LOG_FIELDS = [
    "timestamp", "arm", "status", "model", "arch", "out",
    "dataset", "eval_split", "eval_dataset", "eval2_dataset",
    "r", "alpha", "lr", "scheduler", "warmup_steps", "weight_decay",
    "batch", "grad_accum", "world_size", "eff_batch",
    "epochs", "steps_planned", "steps_done", "seq_len",
    "targets", "compute_dtype", "attn_impl", "parallel", "shuffle",
    "max_samples", "train_embeddings", "train_head", "prompt_format",
    "trainable_params", "n_train", "n_val", "n_eval2",
    "start_loss", "end_loss", "best_val", "best_val_step",
    "final_val", "final_eval2",
    "total_s", "s_per_step", "sup_tok_s", "tot_tok_s", "peak_vram_gb",
    "notes",
]


def append_run_log(path, record):
    """Append one run's metadata as a row to a CSV (header written once on
    create). Pure stdlib so the DDP script imports it and the BNB arm inlines a
    copy. Keys outside RUN_LOG_FIELDS are ignored and missing fields left blank,
    so the column layout is stable run-to-run."""
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in RUN_LOG_FIELDS})
    print(f"[run-log] appended 1 row to {path}")


def turn_end_token(tokenizer):
    """End-of-assistant-turn marker for completion-only SFT, per chat format.

    The model must learn to emit a stop token after the response or generation
    never terminates. The right token is architecture-specific: the Llama-3
    family ends a turn with ``<|eot_id|>``; Mistral/Tekken and most others use
    their EOS (``</s>``). We pick ``<|eot_id|>`` only when it actually exists as
    a special token (preserving the proven Llama path), otherwise the tokenizer's
    EOS. Encoded with ``encode_special_tokens=True`` it maps to the single
    special id, matching the generator's stop condition.
    """
    if "<|eot_id|>" in tokenizer.extended_piece_to_id:
        return "<|eot_id|>"
    if tokenizer.eos_token:
        return tokenizer.eos_token
    return ""


def format_prompt_and_eot(model, tokenizer, prompt_format):
    """Return ``(build_prompt(user) -> str, eot_str)`` for the chosen chat format.

    - ``auto`` (default): the model's own template (``default_chat_prompt`` --
      Llama-3, Mistral ``[INST]``, mistral3 ``[SYSTEM_PROMPT]``/``[INST]``, etc.)
      and the architecture-correct turn-end token (:func:`turn_end_token`).
      Unchanged from prior behavior.
    - ``mistral``: the explicit Mistral V7+/V13 instruct format
      ``<s>[INST]{user}[/INST]{response}</s>`` (no spaces; ``[INST]``/``[/INST]``
      are control tokens). This is what ``auto`` already emits for the ``mistral3``
      arch (Mistral Small/Medium 3.x, incl. Mistral-Medium-3.5-128B) -- the
      explicit option just doesn't depend on arch detection. EOS ends the turn.
    - ``metharme``: the Pygmalion/Metharme format
      ``<s><|user|>{user}<|model|>{response}</s>``. The ``<|user|>``/``<|model|>``
      markers are plain text on a base model (not registered special tokens) --
      the model learns them as a literal pattern, which is the standard way these
      tunes are trained. EOS ends the turn.

    For ``mistral``/``metharme`` a literal BOS is prepended so the sequence starts
    with one; the caller's BOS-normalization then collapses any duplicate the
    tokenizer auto-adds.
    """
    if prompt_format == "mistral":
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        return (lambda user: f"{bos}[INST]{user}[/INST]"), eos
    if prompt_format == "metharme":
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        return (lambda user: f"{bos}<|user|>{user}<|model|>"), eos
    if prompt_format == "auto":
        return (lambda user: model.default_chat_prompt(user)), turn_end_token(tokenizer)
    raise ValueError(f"unknown prompt-format '{prompt_format}' (expected auto/mistral/metharme)")


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


def extract_single_turn(messages):
    """Pull (user_text, assistant_text) from an OpenAI-style ``messages`` list.

    For single-turn rows (e.g. UnstableLlama/semancy: one user, one assistant,
    no system message) this is exact. We take the last user turn that precedes
    the first assistant turn as the prompt and that assistant turn as the
    target, so the completion-only mask still supervises only the answer. System
    messages are ignored (the dataset embeds its reasoning style in the answer
    text rather than a system prompt). Returns ("", "") if either turn is
    missing so the caller can skip the row.
    """
    user_text, asst_text = "", ""
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role == "user":
            user_text = content       # remember the most recent user turn
        elif role == "assistant":
            asst_text = content
            break                     # first assistant reply is the target
    return user_text, asst_text


def make_lr_scheduler(optimizer, name, total_steps, warmup_steps):
    """A transformers-free LR scheduler (none/linear/cosine) with linear warmup.

    Matches HuggingFace's ``get_{linear,cosine}_schedule_with_warmup`` exactly so
    behavior is well understood: LR ramps 0->1 over ``warmup_steps``, then decays
    to 0 (linear) or follows a half-cosine to 0 (cosine) over the remaining
    ``total_steps - warmup_steps``. ``none``/``constant`` holds the base LR after
    warmup. Driven by one ``scheduler.step()`` per optimizer step.
    """
    name = (name or "none").lower()
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if name in ("none", "constant"):
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown scheduler '{name}' (expected none/linear/cosine)")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_steps_and_warmup(args, num_train_examples, effective_batch):
    """Finalize args.steps (from --epochs if given) and compute warmup steps.

    ``--epochs`` (when > 0) overrides ``--steps`` so the schedule length matches
    the requested number of passes over the data: one optimizer step consumes
    ``effective_batch`` examples, so an epoch is ``ceil(N / effective_batch)``
    steps. ``--warmup-steps`` (when > 0) wins over ``--warmup-ratio``.
    """
    if getattr(args, "epochs", 0) and args.epochs > 0:
        eff = max(1, int(effective_batch))
        steps_per_epoch = max(1, math.ceil(num_train_examples / eff))
        args.steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    warmup = (args.warmup_steps if getattr(args, "warmup_steps", 0) and args.warmup_steps > 0
              else int(round(getattr(args, "warmup_ratio", 0.0) * args.steps)))
    return args.steps, max(0, warmup)


def build_sft_examples(model, tokenizer, dataset_name, max_samples, seq_len,
                       instruction_key="instruction", context_key="context",
                       response_key="response", split="train",
                       clean_text=True, min_response_words=3,
                       uppercase_response=False, messages_key=None,
                       prompt_format="auto", shuffle=False, shuffle_seed=0,
                       config_name=None):
    """
    Load an instruction dataset and tokenize for completion-only SFT using the
    model's native chat template (Llama-3, Mistral, etc. -- whatever
    ``model.default_chat_prompt`` emits for this architecture). Prompt tokens are
    masked with -100 so loss is computed only over the (styled) response, which
    is terminated with the architecture-correct turn-end token (see
    :func:`turn_end_token`) so the model learns to stop.

    Two input layouts are supported:
      * flat columns -- instruction_key / context_key / response_key (Alpaca,
        Dolly, ...); context_key may be absent in the dataset (treated as empty).
      * OpenAI ``messages`` -- pass ``messages_key`` (e.g. "messages") for
        single-turn user/assistant rows (UnstableLlama/semancy); the user turn
        becomes the prompt and the assistant turn the supervised response. When
        set it takes precedence over the flat-column keys.

    clean_text strips stage directions / inline actions and normalizes
    whitespace (helps play-script style sets like the Shakespeare default, whose
    raw rows otherwise teach the model to emit "[stage directions]"). Rows whose
    cleaned response has fewer than min_response_words tokens are dropped.

    shuffle (with shuffle_seed) permutes the rows once after loading, BEFORE any
    --val-frac carve and before training, so the held-out split is a random
    sample rather than the first N rows and training order is randomized. It is
    deterministic given the seed, so the EXL3 and BNB arms (which call the same
    HF datasets shuffle) stay matched. Default off preserves the original order;
    the existing shuffle-on-cap (random subset when capping) is unchanged.

    Returns a list of dicts with python int lists: input_ids / labels.
    """
    import os
    from datasets import load_dataset

    # Accept either a Hub dataset id or a local file (e.g. a styled set produced
    # by examples/make_style_dataset.py). load_dataset() can't sniff a bare local
    # path, so pick the builder from the extension when the path exists.
    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        builder = {".json": "json", ".jsonl": "json",
                   ".parquet": "parquet", ".csv": "csv"}.get(ext, "json")
        ds = load_dataset(builder, data_files=dataset_name, split=split)
    elif config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    # Shuffle the full set when asked, or (as before) when capping rows so the
    # subset is random rather than the first max_samples. shuffle_seed defaults to
    # 0, matching the prior cap behavior exactly when --shuffle is off.
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    build_prompt, eot = format_prompt_and_eot(model, tokenizer, prompt_format)

    examples = []
    for ex in ds:
        if messages_key:
            instr, resp = extract_single_turn(ex.get(messages_key))
            ctx = ""
        else:
            instr = (ex.get(instruction_key) or "").strip()
            ctx = (ex.get(context_key) or "").strip()
            resp = (ex.get(response_key) or "").strip()
        if clean_text:
            instr, ctx, resp = (clean_style_text(instr), clean_style_text(ctx),
                                clean_style_text(resp))
        if not resp or len(resp.split()) < min_response_words:
            continue
        if messages_key and not instr:
            continue  # malformed messages row: no user turn to prompt with
        # Smoke test: a maximally dense+consistent transform (every token of every
        # response changes), so there's no low-loss path that ISN'T uppercased and
        # it must surface in generation. Only the response is transformed, so it
        # proves a learned *behavior*, not input echoing.
        if uppercase_response:
            resp = resp.upper()
        user = instr if not ctx else f"{instr}\n\n{ctx}"

        # default_chat_prompt() already includes <|begin_of_text|> and ends with
        # the assistant header, so encode specials and don't add another BOS.
        # Tokenize the prompt and the response SEPARATELY and concatenate, so the
        # prompt/response boundary is exact -- masking by the prompt-string length
        # is vulnerable to tokenizer boundary merges that mis-align the mask.
        prompt_text = build_prompt(user)
        prompt_ids = tokenizer.encode(
            prompt_text, add_bos=False, encode_special_tokens=True
        )[0].tolist()
        resp_ids = tokenizer.encode(
            resp + eot, add_bos=False, encode_special_tokens=True
        )[0].tolist()

        # Normalize BOS. With encode_special_tokens=True the underlying HF
        # tokenizer adds <|begin_of_text|> itself (Llama-3 has add_bos_token=true),
        # *in addition to* the literal one default_chat_prompt() embeds and *again*
        # on the separately-encoded response -- so the prompt would start with two
        # BOS and the response with a spurious one. Standard Llama-3 (and the BNB
        # arm, which uses add_special_tokens=False) is exactly one BOS at the very
        # start and none mid-sequence. Drop the duplicates; no-op for tokenizers
        # that don't auto-prepend BOS.
        bos = tokenizer.bos_token_id
        if bos is not None:
            while len(prompt_ids) >= 2 and prompt_ids[0] == bos and prompt_ids[1] == bos:
                prompt_ids = prompt_ids[1:]
            if resp_ids and resp_ids[0] == bos:
                resp_ids = resp_ids[1:]

        input_ids = (prompt_ids + resp_ids)[:seq_len]
        labels = [-100] * len(prompt_ids) + list(resp_ids)
        labels = labels[:seq_len]
        if all(l == -100 for l in labels):
            continue  # response got truncated away; skip
        examples.append({"input_ids": input_ids, "labels": labels})

    return examples


def build_lm_examples(tokenizer, dataset_name, split, seq_len,
                      text_key="text", max_samples=0, config_name=None):
    """Plain-text language-modeling eval set (e.g. wikitext) for a second,
    task-independent held-out loss.

    Concatenates the dataset's text column and packs it into non-overlapping
    ``seq_len`` blocks with every token supervised (no completion mask), so the
    resulting loss is a straight nats/token cross-entropy -- on the same scale as
    the SFT eval loss, which lets you watch the two move together (does the task
    fit track or diverge from general LM ability?). Tokenization matches the SFT
    path's underlying tokenizer, so the EXL3 and BNB arms produce identical
    blocks and hence a comparable number.

    Returns a list of dicts (input_ids / labels), same shape as
    :func:`build_sft_examples`, so the same eval loop consumes it.
    """
    import os
    from datasets import load_dataset

    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        builder = {".json": "json", ".jsonl": "json",
                   ".parquet": "parquet", ".csv": "csv"}.get(ext, "json")
        ds = load_dataset(builder, data_files=dataset_name, split=split)
    elif config_name:
        # Many text corpora need a config (e.g. wikitext -> "wikitext-2-raw-v1").
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    bos = tokenizer.bos_token_id
    buf, examples = [], []
    for row in ds:
        text = row.get(text_key) or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_bos=False,
                               encode_special_tokens=False)[0].tolist()
        # The tokenizer may auto-prepend BOS; drop it so packing doesn't inject a
        # BOS at every row boundary (mid-stream BOS would distort the LM loss).
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        buf.extend(ids)
        while len(buf) >= seq_len:
            block = buf[:seq_len]
            buf = buf[seq_len:]
            examples.append({"input_ids": block, "labels": list(block)})
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


def sample(model, cache, tokenizer, generator, build_prompt, prompt, max_new_tokens=48):
    """Quick native generation for live progress feedback (uses the same chat
    format as training, so a metharme-trained adapter previews meaningfully)."""
    text = build_prompt(prompt)
    resp = generator.generate(
        prompt=text, max_new_tokens=max_new_tokens,
        add_bos=False, completion_only=True,
    )
    return resp.strip().replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    ap.add_argument("--device", default="cuda:0",
                    help="single-device load target (ignored when --parallel split)")
    ap.add_argument("--parallel", choices=["single", "split"], default="single",
                    help="single: one GPU; split: layer-autosplit the frozen base "
                         "across visible GPUs (memory, for models too big for one card)")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB to reserve per device; negative excludes a device")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB budget per device; caps a card to force/tune the split")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the LoRA params (default 0.01).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"],
                    default="none",
                    help="LR schedule after warmup: none (constant), linear "
                         "decay to 0, or cosine decay to 0.")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Fraction of total steps spent linearly warming up the "
                         "LR from 0 (e.g. 0.05-0.1). Ignored if --warmup-steps>0.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when >0.")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, set --steps to cover this many passes over the "
                         "training data (one step = batch*grad-accum examples), "
                         "so the schedule length matches the epoch count.")
    ap.add_argument("--steps", type=int, default=1000,
                    help="Training steps (ignored when --epochs>0). ~steps*batch "
                         "examples seen; aim for >=1 epoch to pick up a style.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--dataset",
        default="superdrew100/UwU_Alpaca_data",
        help="HF dataset id. Default is the UwU-furry Alpaca style set.",
    )
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction",
                    help="Column holding the prompt/instruction")
    ap.add_argument("--context-key", default="input",
                    help="Optional extra-context column; absent columns are ignored "
                         "(Alpaca uses 'input', Dolly uses 'context')")
    ap.add_argument("--response-key", default="output",
                    help="Column holding the target response (Alpaca: 'output', "
                         "Dolly: 'response')")
    ap.add_argument("--messages-key", default=None,
                    help="Column holding OpenAI-style single-turn messages (e.g. "
                         "'messages' for UnstableLlama/semancy). When set, the "
                         "user turn is the prompt and the assistant turn the "
                         "supervised response; --instruction/context/response-key "
                         "are ignored.")
    ap.add_argument("--prompt-format", choices=["auto", "mistral", "metharme"],
                    default="auto",
                    help="Chat format. auto: the model's native template "
                         "(Llama-3, Mistral [INST], mistral3 [SYSTEM_PROMPT]/[INST]). "
                         "mistral: explicit <s>[INST]{q}[/INST]{a}</s> (= auto for "
                         "the mistral3 arch, e.g. Mistral-Medium-3.5). metharme: "
                         "Pygmalion <|user|>{q}<|model|>{a}</s>. EOS ends the turn "
                         "for mistral/metharme.")
    ap.add_argument("--no-clean-text", action="store_true",
                    help="Disable stripping of [stage directions]/*actions* and "
                         "whitespace normalization (on by default; helps play-script "
                         "style sets, leave off for code/markdown datasets)")
    ap.add_argument("--min-response-words", type=int, default=3,
                    help="Drop rows whose cleaned response is shorter than this")
    ap.add_argument("--uppercase-response", action="store_true",
                    help="Smoke test: train the model to RESPOND IN ALL CAPS. A "
                         "maximally dense/consistent transform that must show in "
                         "generation if the training path works at all.")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the training rows once (deterministically) "
                         "before the --val-frac carve and before training, so the "
                         "held-out split is a random sample and training order is "
                         "randomized. Matched across arms given the same seed.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for --shuffle (also the random-subset seed when "
                         "--max-samples caps the rows). Default 0.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="Tokenization check: decode the first N built examples "
                         "(prompt span vs supervised response span + whether the "
                         "response was truncated by --seq-len), then exit without "
                         "training. Run this once to verify a new dataset/schema.")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target module leaf names (default: attn+mlp projections)")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="Also FULLY train the input embeddings (modules_to_save), "
                         "not just LoRA. Saved to modules_to_save.safetensors. Big "
                         "(vocab x hidden) -- raises VRAM and, under DDP, the "
                         "per-step grad all-reduce. On a tied model this also "
                         "trains the head (shared weight).")
    ap.add_argument("--train-head", action="store_true",
                    help="Also FULLY train the LM head (modules_to_save). Switches "
                         "the loss off the fused frozen-head path to a supervised-"
                         "position cross-entropy so the head gets a gradient. On a "
                         "tied model this is equivalent to --train-embeddings.")
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="Attention kernel: auto (FlashAttention-2 when the "
                         "flash_attn package is importable and the run is CUDA "
                         "fp16/bf16, else eager), flash (require it), or eager "
                         "(the reference; O(t^2) memory). Flash is O(t) memory -- "
                         "the lever for long-context training.")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=25,
                    help="Generate a sample completion every N steps (0 to disable)")
    ap.add_argument("--sample-prompt", default="Tell me about your day.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Checkpoint the adapter to --out every N steps (0 = only "
                         "at the end). The adapter is also saved on Ctrl-C.")
    ap.add_argument("--resume", default=None,
                    help="Adapter dir to resume from (continues training those "
                         "weights; optimizer state is NOT restored). --r/--targets "
                         "must match the checkpoint.")
    ap.add_argument("--eval-split", default=None,
                    help="Use this split of the dataset (e.g. 'test') as the "
                         "held-out eval set, instead of carving --val-frac off "
                         "train. Real held-out data; takes precedence over "
                         "--val-frac.")
    ap.add_argument("--eval-dataset", default=None,
                    help="Dataset id/path for --eval-split (defaults to --dataset).")
    ap.add_argument("--eval2-dataset", default=None,
                    help="A SECOND held-out eval set, reported alongside the "
                         "primary one each --eval-every and at the end, so you can "
                         "watch them move together (e.g. your test set vs "
                         "wikitext). --save-best stays keyed on the PRIMARY eval.")
    ap.add_argument("--eval2-split", default="test",
                    help="Split for --eval2-dataset (default 'test').")
    ap.add_argument("--eval2-config", default=None,
                    help="HF dataset config for --eval2-dataset (e.g. "
                         "'wikitext-2-raw-v1' for the 'wikitext' dataset).")
    ap.add_argument("--eval2-text-key", default=None,
                    help="If set, treat --eval2-dataset as PLAIN TEXT and compute "
                         "a language-modeling loss over packed --seq-len blocks "
                         "(every token supervised) -- e.g. 'text' for wikitext. "
                         "If unset, --eval2-dataset is built as a second SFT eval "
                         "using the same instruction/messages keys.")
    ap.add_argument("--eval2-max-samples", type=int, default=0,
                    help="Cap source rows for --eval2-dataset (0 = all).")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="Hold out this fraction of train for held-out eval loss "
                         "(deterministic; the SAME split as qlora_train_bnb.py "
                         "given the same dataset/seed). Ignored if --eval-split is "
                         "set. 0 = no eval.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Also report held-out loss every N steps (needs "
                         "--val-frac > 0). 0 = only at the end.")
    ap.add_argument("--save-best", action="store_true",
                    help="Save the adapter only when held-out loss improves "
                         "(needs --val-frac + --eval-every), so a long run keeps "
                         "the best checkpoint instead of an overfit endpoint.")
    ap.add_argument("--run-log", default="qlora_runs.csv",
                    help="Append one metadata row per run to this CSV (model, "
                         "hyperparameters, start/end/best-val loss, timing, tok/s, "
                         "peak VRAM, ...). Written on normal finish AND on Ctrl-C. "
                         "Empty string disables.")
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

    if args.parallel == "split":
        load_kwargs = {}
        if args.reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = args.reserve_per_device
        if args.use_per_device is not None:
            load_kwargs["use_per_device"] = args.use_per_device
        model.load(progressbar=True, **load_kwargs)
        active_devices = list(model.active_devices)
        print(f" -- layer-autosplit: active devices {active_devices}, "
              f"output device {model.output_device}")
    else:
        model.load(device=args.device, progressbar=True)
        active_devices = [torch.device(args.device).index]
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 2. Build the differentiable QLoRA model (frozen base + trainable adapters).
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        compute_dtype=cdt, gradient_checkpointing=not args.no_grad_ckpt,
        train_embeddings=args.train_embeddings, train_head=args.train_head,
        attn_impl=args.attn_impl,
    )
    net.train()
    if args.resume:
        net.load_adapter(args.resume)
    ms = [n for n, p in [("embed", net.embed_weight), ("head", net.head_weight)] if p is not None]
    print(f" -- trainable params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules}"
          f"{', modules_to_save=' + str(ms) if ms else ''})")
    print(f" -- {net.describe_attn()}")
    if args.parallel == "split":
        from collections import Counter
        dist = Counter(str(d) for d in net._block_devices)
        print(f" -- decoder block devices: {dict(dist)}  (final norm + head on {net.device})")

    # 3. Data.
    examples = build_sft_examples(
        model, tokenizer, args.dataset, args.max_samples, args.seq_len,
        instruction_key=args.instruction_key, context_key=args.context_key,
        response_key=args.response_key, split=args.dataset_split,
        clean_text=not args.no_clean_text,
        min_response_words=args.min_response_words,
        uppercase_response=args.uppercase_response,
        messages_key=args.messages_key,
        prompt_format=args.prompt_format,
        shuffle=args.shuffle, shuffle_seed=args.shuffle_seed,
    )
    print(f" -- {len(examples)} SFT examples{' (shuffled)' if args.shuffle else ''}")
    assert examples, "no usable training examples"

    # Tokenization check: decode the prompt span (labels==-100) and the supervised
    # response span (labels!=-100) separately, so the mask boundary and any
    # --seq-len truncation are visible before committing to a run. Specials are
    # shown so the chat template / <|eot_id|> stop token can be eyeballed.
    if args.inspect:
        _, eot = format_prompt_and_eot(model, tokenizer, args.prompt_format)
        eot_id = tokenizer.encode(eot, add_bos=False,
                                  encode_special_tokens=True)[0].tolist() if eot else []
        # encode() auto-prepends BOS (see build_sft_examples); strip it so the
        # "ends with turn-end token" check compares the real eot id(s), not [BOS, eot].
        bos = tokenizer.bos_token_id
        if eot_id and bos is not None and eot_id[0] == bos:
            eot_id = eot_id[1:]
        for i, ex in enumerate(examples[:args.inspect]):
            ids, labs = ex["input_ids"], ex["labels"]
            n_prompt = sum(1 for l in labs if l == -100)
            sup = [t for t, l in zip(ids, labs) if l != -100]
            prompt_ids = ids[:n_prompt]
            dec = lambda seq: tokenizer.decode(torch.tensor([seq]),
                                               decode_special_tokens=True)
            ends_eot = bool(eot_id) and sup[-len(eot_id):] == eot_id
            print(f"\n===== example {i} | {len(ids)} tokens "
                  f"({n_prompt} prompt / {len(sup)} supervised) =====")
            print(f"  PROMPT  (masked, -100): {dec(prompt_ids)!r}")
            print(f"  RESPONSE(supervised)  : {dec(sup)!r}")
            print(f"  ends with turn-end token ({eot!r})? {ends_eot}"
                  + ("" if ends_eot else
                     "   <-- WARNING: response truncated by --seq-len; "
                     "raise --seq-len so the model learns to stop"))
        print(f"\n -- inspect only ({args.inspect} shown); exiting before training.")
        return

    # Held-out eval set. Prefer the dataset's own eval split (real held-out data);
    # otherwise carve a deterministic val_frac off the front of train (same rows
    # as qlora_train_bnb.py, so the arms' eval losses stay comparable).
    val_examples = []
    if args.eval_split:
        val_examples = build_sft_examples(
            model, tokenizer, args.eval_dataset or args.dataset, 0, args.seq_len,
            instruction_key=args.instruction_key, context_key=args.context_key,
            response_key=args.response_key, split=args.eval_split,
            clean_text=not args.no_clean_text,
            min_response_words=args.min_response_words,
            uppercase_response=args.uppercase_response,
            messages_key=args.messages_key,
            prompt_format=args.prompt_format,
        )
        print(f" -- held-out eval: {len(val_examples)} examples from "
              f"split '{args.eval_split}'; {len(examples)} for training")
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val examples; "
              f"{len(examples)} for training")
        assert examples, "val_frac too large; no training examples left"

    # Optional SECOND held-out eval set (task-independent monitor, e.g. wikitext).
    # Plain-text LM loss when --eval2-text-key is given, else a second SFT eval.
    val2_examples = []
    eval2_label = ""
    if args.eval2_dataset:
        eval2_label = args.eval2_dataset.split("/")[-1]
        if args.eval2_text_key:
            val2_examples = build_lm_examples(
                tokenizer, args.eval2_dataset, args.eval2_split, args.seq_len,
                text_key=args.eval2_text_key, max_samples=args.eval2_max_samples,
                config_name=args.eval2_config)
            kind = f"LM blocks over '{args.eval2_text_key}'"
        else:
            val2_examples = build_sft_examples(
                model, tokenizer, args.eval2_dataset, args.eval2_max_samples,
                args.seq_len, instruction_key=args.instruction_key,
                context_key=args.context_key, response_key=args.response_key,
                split=args.eval2_split, clean_text=not args.no_clean_text,
                min_response_words=args.min_response_words,
                uppercase_response=args.uppercase_response,
                messages_key=args.messages_key, prompt_format=args.prompt_format,
                config_name=args.eval2_config)
            kind = "SFT"
        print(f" -- eval2 ({eval2_label}): {len(val2_examples)} {kind} examples "
              f"from split '{args.eval2_split}'")

    # Finalize step count (from --epochs) and warmup before building the schedule.
    args.steps, warmup_steps = resolve_steps_and_warmup(
        args, len(examples), args.batch * args.grad_accum)
    print(f" -- {args.steps} steps, scheduler={args.scheduler}, "
          f"warmup={warmup_steps}, weight_decay={args.weight_decay}")

    # 4. Optional generator for live samples (KV-cache inference path). The cache
    #    was allocated before load() above. Use the training chat format so the
    #    preview is meaningful for a metharme-trained adapter.
    build_prompt, _ = format_prompt_and_eot(model, tokenizer, args.prompt_format)
    generator = None
    if args.sample_every:
        from exllamav3 import Generator
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
        net.eval()
        with torch.inference_mode():
            base = sample(model, cache, tokenizer, generator, build_prompt, args.sample_prompt)
        net.train()
        print(f"\n\U0001f3ad  baseline (step 0): {args.sample_prompt}\n     -> {base}\n")

    # 5. Optimizer over the trainable params, plus the LR schedule. Weight decay
    #    on the LoRA params only (param_groups puts embed/head in a 0-WD group).
    opt = torch.optim.AdamW(net.param_groups(args.weight_decay), lr=args.lr)
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(0).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    def adapter_b_norm():
        # Per-wrapper sums live on each wrapper's own device (they differ under a
        # layer split), so reduce each to a Python float before summing -- adding
        # tensors across cuda:0/cuda:1 would raise a cross-device error.
        with torch.no_grad():
            return sum(w.lora_b.float().pow(2).sum().item() for w in net._wrappers
                       if w.r > 0) ** 0.5

    def save(tag):
        # Always leave net in train mode after; saving touches the adapter only.
        net.save_adapter(args.out, base_model_name_or_path=args.model)
        print(f"{tag} Adapter written to {args.out}")

    def eval_loss(exs):
        # Mean per-example loss over an eval set, one example at a time (no
        # padding effects). qlora_train_bnb.py computes this identically. Works
        # for both SFT (completion-masked) and plain-LM (all-supervised) sets.
        if not exs:
            return None
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ex in exs:
                input_ids, labels, attn = collate([ex], pad_id)
                l = net.compute_loss(input_ids, labels, attention_mask=attn,
                                     chunk=args.ce_chunk)
                total += l.item()
                n += 1
        net.train()
        return total / n

    def evaluate():
        return eval_loss(val_examples)

    bgen = batches()
    opt.zero_grad(set_to_none=True)
    ema = None
    step = 0
    best_val = float("inf")
    best_val_step = 0
    start_loss = end_loss = None
    last_eval_step, last_val, last_eval2 = -1, None, None
    tok_seen, tot_seen, t0 = 0, 0, time.time()
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    meter = ThroughputMeter()
    if torch.cuda.is_available():
        for d in active_devices:
            torch.cuda.reset_peak_memory_stats(d)

    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return max((torch.cuda.max_memory_allocated(d) / 1e9 for d in active_devices),
                   default=0.0)

    def log_run(status, dt, final_val, final_eval2):
        # One CSV row capturing the run's identity, hyperparameters and results.
        # Called on normal finish and on Ctrl-C so interrupted runs are recorded.
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "exl3-native", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "", "eval_dataset": args.eval_dataset or "",
            "eval2_dataset": args.eval2_dataset or "",
            "r": args.r, "alpha": args.alpha, "lr": args.lr,
            "scheduler": args.scheduler, "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay, "batch": args.batch,
            "grad_accum": args.grad_accum, "world_size": 1,
            "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(net.target_modules), "compute_dtype": args.compute_dtype,
            "attn_impl": args.attn_impl, "parallel": args.parallel,
            "shuffle": int(bool(args.shuffle)), "max_samples": args.max_samples,
            "train_embeddings": int(bool(args.train_embeddings)),
            "train_head": int(bool(args.train_head)), "prompt_format": args.prompt_format,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples), "n_eval2": len(val2_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "final_val": rnd(final_val), "final_eval2": rnd(final_eval2),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(tok_seen / dt) if dt else "",
            "tot_tok_s": round(tot_seen / dt) if dt else "",
            "peak_vram_gb": rnd(peak_vram_gb(), 3), "notes": "",
        })
    try:
        for step in range(1, args.steps + 1):
            step_t0 = time.time()
            accum_loss = 0.0
            step_sup = step_tot = 0
            for _ in range(args.grad_accum):
                batch = next(bgen)
                input_ids, labels, attn = collate(batch, pad_id)
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk)
                (loss / args.grad_accum).backward()
                accum_loss += loss.item() / args.grad_accum
                step_sup += int((labels != -100).sum())   # supervised tokens
                step_tot += int(attn.sum())                # total (non-pad) tokens

            # grad norm BEFORE clipping is a direct check that gradients reach the
            # adapters (a flat ~0 here would mean the backward graph is broken).
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.trainable_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            # Rolling tok/s over the train-step compute only (eval/sample/save
            # below are excluded so the rate reflects steady-state throughput).
            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            _, tot_tps = meter.rates()

            if start_loss is None:
                start_loss = accum_loss
            end_loss = accum_loss
            ema = accum_loss if ema is None else 0.9 * ema + 0.1 * accum_loss
            print(f"  step {step:>5}/{args.steps} | loss {accum_loss:6.4f} | "
                  f"ema {ema:6.4f} | grad {gnorm:7.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"|B| {adapter_b_norm():7.3f} | {tot_tps:,.0f} tok/s")

            if (args.eval_every and step % args.eval_every == 0
                    and (val_examples or val2_examples)):
                vl = evaluate() if val_examples else None
                v2 = eval_loss(val2_examples) if val2_examples else None
                last_eval_step, last_val, last_eval2 = step, vl, v2
                parts = []
                if vl is not None:
                    parts.append(f"held-out {vl:.4f}")
                    # Track best val for the run log regardless of --save-best;
                    # only write the checkpoint when --save-best is set.
                    if vl < best_val:
                        best_val = vl
                        best_val_step = step
                        if args.save_best:
                            save(f"[best step {step}, val {vl:.4f}]")
                if v2 is not None:
                    parts.append(f"{eval2_label} {v2:.4f}")
                print(f"    [eval] step {step}: " + " | ".join(parts))

            if args.sample_every and step % args.sample_every == 0:
                net.eval()
                net.apply_to_native()      # make generation reflect the adapter
                with torch.inference_mode():
                    txt = sample(model, cache, tokenizer, generator, build_prompt, args.sample_prompt)
                net.remove_from_native()
                net.train()
                print(f"\n  \U0001f3ad  [step {step}] {args.sample_prompt}\n     -> {txt}\n")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")
    except KeyboardInterrupt:
        # Stopping early at the loss plateau is a normal workflow; don't discard
        # the adapter trained so far -- unless --save-best already kept the
        # best-val checkpoint, in which case saving now would clobber it with
        # later (likely overfit) weights.
        if args.save_best and val_examples:
            print(f"\nInterrupted at step {step}; keeping best-val adapter.")
        else:
            print(f"\nInterrupted at step {step}; saving adapter before exit.")
            if step > 0:
                save("[interrupted]")
        log_run("interrupted", time.time() - t0, None, None)
        raise SystemExit(0)

    # 6. Save adapter (PEFT format; loadable by exllamav3.model.lora.LoRA).
    #    With --save-best we already kept the best-val checkpoint; don't clobber
    #    it with the (likely overfit) final-step weights.
    dt = time.time() - t0
    # Final held-out numbers. Reuse the last in-loop eval when it already ran on
    # the final step (avoids a duplicate full pass that looks like a hang after
    # "Done."); otherwise compute once, announcing it so the GPU churn is expected.
    if last_eval_step == step:
        val_loss, final_eval2 = last_val, last_eval2
    elif val_examples or val2_examples:
        print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss = evaluate()
        final_eval2 = eval_loss(val2_examples) if val2_examples else None
    else:
        val_loss, final_eval2 = None, None
    if not (args.save_best and val_examples):
        save("Done.")
    if val_loss is not None:
        tag = f" (best kept: {best_val:.4f})" if args.save_best else ""
        print(f"\n[EVAL] held-out loss (EXL3 arm): {val_loss:.4f}{tag} "
              f"over {len(val_examples)} examples")
    if final_eval2 is not None:
        print(f"[EVAL] eval2 ({eval2_label}) loss: {final_eval2:.4f} "
              f"over {len(val2_examples)} examples")
    if torch.cuda.is_available():
        peak_str = " / ".join(
            f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1e9:.2f}GB"
            for d in active_devices
        )
    else:
        peak_str = "n/a"
    print(f"[PERF] {tok_seen / dt if dt else 0:,.0f} sup tok/s, "
          f"{tot_seen / dt if dt else 0:,.0f} tot tok/s | "
          f"peak VRAM {peak_str} | {dt:.0f}s for {step} steps")
    log_run("completed", dt, val_loss, final_eval2)
    print("Verify with: python examples/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}")


if __name__ == "__main__":
    main()
