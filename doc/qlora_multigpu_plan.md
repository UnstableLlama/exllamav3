# QLoRA-on-EXL3 — Multi-GPU plan (`--parallel`)

> Plan doc. Branch: `claude/trusting-goodall-in70lc`. Goal: two selectable
> multi-GPU training paths behind one flag — **DDP** (throughput, models that fit
> one card) and **layer-split** (memory, models that don't), the latter also
> running each card at ~50% duty so they run cooler on multi-day runs.

## Motivation

A 14B 4bpw base nearly fills a 24GB card at batch 2. The dominant costs:
base weights ~7.5GB, **LoRA + AdamW state ~5GB at r=64** (≈300M trainable params
× 4 for master/grad/m/v), fp32 residual checkpoints ~1GB, transient dense-weight
reconstructions incl. the 131k-vocab head ~1.3GB. DDP (already built) *replicates*
all of this per card, so it does nothing for the memory wall. **Layer-split**
halves the base AND splits the optimizer/activations across cards.

With two cards the modes are mutually exclusive (split *or* replicate); a hybrid
(split × DDP) only pays off at ≥4 GPUs. Design leaves room for it but doesn't
build it.

## How exllamav3 handles GPU split natively (the reference)

We reuse exllamav3's own layer-split machinery rather than invent placement.

- **Load:** `model.load(reserve_per_device=[...])` (no fixed `device`) →
  `_load_autosplit` (`model/model.py`) distributes layers across GPUs, sets each
  `Module.device`, records `model.active_devices` and
  `model.output_device = modules[-1].device`. `model.load(device="cuda:0")` is the
  single-device path (`_load_single`) the current scripts use.
  - `reserve_per_device` / `use_per_device` (GB, per device; negative reserve
    excludes a device) tune the balance. The head (131k vocab) is heavy on the
    last device, the embedding is on CPU (`prefer_cpu`), so a little reserve
    tuning keeps the split even.
- **Forward:** `forward_ls` (`model/model_ls.py:241`) is a plain sequential loop:
  ```
  for module … :
      x = module.prepare_for_device(x, params)
      x = module.forward(x, params)
  ```
- **Cross-device handoff:** `Module.prepare_for_device` (`modules/module.py:71`)
  moves `x` to `self.device`; under `no_p2p_copy` (env `EXLLAMA_NO_P2P_COPY`,
  `modules/module.py:12`) it **bounces through CPU** (`x.cpu().to(device)`) for
  systems without GPU peer access — relevant to the PCIe ×4 card here.
- **TP/tensor-parallel** (`tensor_p=True` → `forward_tp`, `model/model_tp_fn.py`,
  worker processes) is a separate, heavier mode. **Out of scope.** "split" here
  means *layer* split (the alternating, cooler mode).

**Implication for training:** mirror `forward_ls` — migrate the hidden state (and
`position_ids` / `attn_bias`) to each block's device at the boundary, using the
same `no_p2p_copy`-aware logic. Native `forward_ls` is `@torch.inference_mode`;
ours can't be, but `.to()`/`.cpu()` are autograd-friendly so gradients flow back
across the boundary for free.

## Design

### Flag & launch
One `--parallel {single,ddp,split}`; a thin launcher (shell wrapper) picks the
process model because the two genuinely differ:

| `--parallel` | launcher | behavior |
|---|---|---|
| `single` (default) | `python` | one GPU |
| `ddp` | `torchrun --nproc_per_node=N` | replicate per rank, shard batch, all-reduce LoRA grads |
| `split` | `python` | one process, base layer-autosplit across visible GPUs |

### `ParallelContext` abstraction
The training loop talks to one object, three implementations, so no scattered
`if parallel == …`:
- `devices` — single: one; ddp: the rank's; split: `model.active_devices`
- `is_main` — rank-0 for ddp, else True
- `shard(dataset)` — ddp shards by rank; single/split return all
- `all_reduce_grads(params)` / `all_reduce_scalar(x)` — real NCCL under ddp;
  **no-ops** under single/split
- `barrier()` — no-op except ddp

### Device-aware forward (the one piece of genuinely new model code)
- `backbone.block_device(block)` → `block.device`; `backbone.to_device(x, dev)`
  mirrors `prepare_for_device` (honors `no_p2p_copy`).
- `NativeLlamaQLoRA`: replace the single `self.device` assumption with a per-block
  device list. In `forward()`, migrate `hidden`/`position_ids`/`attn_bias` to the
  block's device when it changes; final norm + head on the last device; the
  fused-CE head matmul runs on the head's device.
- **Adapters need no change** — `DiffLinear` already inits `lora_a/lora_b` on
  `backbone.linear_device(linear)`, so under autosplit they land on the right card.
- Single-device is preserved exactly: all devices equal ⇒ every migration is a
  no-op, identical to today.

### Model loading per mode
- single / ddp: `model.load(device=ctx.device)` (unchanged).
- split: `model.load(reserve_per_device=[...])` (autosplit). Expose
  `--reserve-per-device` passthrough for balance tuning.

### Shared-loop extraction
Move `build_sft_examples`, masking, `collate`, `evaluate`, `save`, and the step
loop into `training/qlora_sft_common.py`, imported by the entry script — kills the
byte-for-byte duplication between the single and DDP scripts (a third copy would
be worse). DDP-specific bits live behind the `ParallelContext`.

## Sequencing
1. **Device-aware forward + `backbone.block_device`/`to_device`** — the
   load-bearing piece; preserves single-device, CPU-regression-safe.
   **DONE & validated on 2×3090** (2026-06): `qlora_validate_native.py --parallel
   split` on Llama-3.2-1B forced to a 7/9 layer split — forward matches native
   (100% argmax, cos 0.999999) and `--check-backward` confirms cross-device
   gradient flow (grads on both cards) through checkpointing. Single-device
   unchanged; all CPU suites green.
2. `ParallelContext` (single/ddp/split). **DEFERRED** (maintainability only).
3. Shared-loop extraction + entry refactor (`--parallel`). **PARTIAL / DEFERRED:**
   split support was added directly to `qlora_train_native.py` as "increment A"
   (`--parallel single|split` + `--reserve/--use-per-device`, per-card VRAM
   report) and **validated on 2×3090** — a 20-step 1B split run trained cleanly
   (loss 2.72→1.58, memory split cuda:0 0.89 / cuda:1 4.42 GB). The shared-loop
   extraction into `training/qlora_sft_common.py` + `ParallelContext` (the dedup,
   steps 2–3) is **not started** — touches both working scripts, no functional
   gain; do it carefully in CPU-suite-checkable pieces if/when wanted.
4. Wrapper dispatch + flip `train_rocinante_yoda.sh` default to `--parallel split`.
   **DONE:** `PARALLEL=split|ddp` selects launcher; split is the default; the
   validate gate runs under the same split args; the greedy-fill footgun (cap
   cuda:0 via `USE_PER_DEVICE`) is documented in the script header.

## Tests
- CPU regression: the existing `tests/test_native_llama.py` exercises
  `_block_forward`/`DiffLinear` on a single (cpu) device and must stay green
  (the device-aware paths are no-ops there).
- Box validation: forward-correctness gate (`qlora_validate_native.py`) under a
  split load, then a short split run vs a single-GPU loss curve at matched
  effective batch.

## Notes / gotchas
- ddp + split don't compose on 2 cards (room left for a `hybrid` mode later).
- Live `sample()` stays on under split (native generator handles the autosplit
  cache), off under ddp — as today.
- Balance the split by *memory*, not layer count (head/embedding are end-heavy).
