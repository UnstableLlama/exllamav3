# QLoRA training pipeline follow-up checklist

This checklist tracks the training-pipeline review items from the YAML/QLoRA
cleanup pass. It is intentionally split into small work items so future sessions
can land and validate them independently.

## Completed in this pass

- [x] **Remove duplicate run-log helpers in the trainers.**
  - `examples/qlora_train_native.py` and `examples/qlora_train_native_ddp.py` had
    duplicate `log_run` helper blocks left by earlier merge work. Keep one copy so
    baseline eval fields and new eval cap fields are logged consistently.
- [x] **Make the micro-batch sampler robust for small datasets and tails.**
  - The old sampler skipped tail rows and could spin forever when a dataset/shard
    was smaller than `--batch`. The sampler now wraps rows to keep full
    micro-batches, which preserves stable batch shape while avoiding hangs.
- [x] **Default live samples off in the YAML/single trainer.**
  - `sample_every` now defaults to `0`, avoiding the live-generation KV-cache VRAM
    cost unless the user opts in.
- [x] **Log eval cap knobs.**
  - Add `eval_max_samples`, `eval2_max_samples`, and `eval2_max_blocks` to the CSV
    run-log schema and payloads.
- [x] **Clean the DDP checklist footer.**
  - The duplicated note at the bottom of `qlora_train_native_ddp.py` is gone.
- [x] **Chunk the trainable-head / LoRA-head CE path.**
  - The materialized supervised-position loss path now computes logits in token
    chunks, preserving mean CE semantics while bounding peak logit memory to
    `token_chunk × vocab`.
  - Added a CPU numerical check against the naive full-logit path, including
    gradients for hidden states, head weights, and head-LoRA parameters.

## High-priority follow-up work

- [ ] **Make DDP eval rank-0-only plus broadcast.**
  - DDP currently evaluates the full held-out set on every rank. Rank 0 can compute
    the scalar losses and broadcast them so all ranks branch identically for
    `--save-best`, cutting eval compute by roughly `world_size`.
- [ ] **Coalesce or bucket DDP gradient all-reduces.**
  - The DDP trainer manually all-reduces each trainable parameter one at a time.
    Coalescing LoRA gradients by device/dtype, or moving to a real DDP wrapper if
    feasible, should reduce latency and improve scaling.
- [ ] **Add optional batched eval.**
  - Current eval is batch-1 for clean per-example losses. Add an `eval_batch_size`
    knob while preserving the existing mean-per-example metric semantics.

## Medium-priority polish

- [ ] **Mirror more single/split memory levers into DDP where feasible.**
  - Candidate features: `use_liger`, `offload_activations`, `lora_embed`,
    `lora_head`, and 8-bit LoRA optimizer support.
- [ ] **Add targeted tests for YAML/train-loop edge cases.**
  - Suggested cases: `targets: []` dry-run, `eval_max_samples` forwarding,
    unknown nested `ddp:` key failure, micro-batch sampler with `len(data) < batch`,
    and run-log schema field preservation.
- [ ] **Consider logging `eval_batch_size` once that knob exists.**
  - Keep the CSV useful for reproducing eval speed/metric choices.

## Notes

- DDP eval and DDP all-reduce improvements should be separate PRs so performance
  changes are easy to reason about and benchmark.
