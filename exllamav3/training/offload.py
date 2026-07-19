"""
Async double-buffered CPU activation offload (S36, Tier-2 item 2).

``torch.autograd.graph.save_on_cpu(pin_memory=True)`` — the S9 offload this
replaces — issues a BLOCKING ``cudaMemcpy`` per saved tensor, in both
directions: every grad-checkpointed block boundary drains the launch pipeline
once on the way out (forward, DtoH) and once on the way back (backward, HtoD).
The S35 profile put that stall at ~162 ms/step on the Llama-3B run class —
the single largest fixable bucket, and exactly why unsloth's checkpointed
offload looks free: they overlap the copies with compute.

This module does the same with plain CUDA streams, no Triton:

  * pack (forward): the saved tensor is copied to a PINNED host buffer on a
    per-device side stream. The compute stream never waits; the DMA engine
    runs the copy while the next block computes. ``record_stream`` keeps the
    source block's memory from being reused until the copy has read it —
    VRAM is released exactly as save_on_cpu released it, one block later.
  * unpack (backward): copies back HtoD on the same side stream, and
    PREFETCHES the next-to-be-unpacked records (autograd consumes them in
    reverse pack order) so block i-1's activation is already on device while
    block i recomputes. The compute stream only ``wait_event``s — GPU-side
    ordering, no host sync.
  * pinned buffers are pooled by (shape, dtype) and reused across steps —
    ``cudaHostAlloc`` is itself synchronizing, so save_on_cpu's per-call
    pinned allocation was part of the stall. Steady state holds one buffer
    per saved tensor (~n_blocks x [b, t, d]; ~700 MB host RAM on the 3B
    reference config, the same amount save_on_cpu allocated transiently).

The pool is BOUNDED (S47). The original pool was keyed by (shape, dtype) with
no cap and no eviction, on the assumption above that "steady state holds one
buffer per saved tensor". That assumption only holds for SHAPE-STABLE runs
(packed SFT). Any run whose per-step activation shape varies — EBFT (random
anchors → different padded rollout length each step) or *unpacked* SFT
(variable real sequence length each step) — mints a fresh (shape, dtype) key
every step and orphans the previous step's buffers in the pool forever, so
pinned host RAM grows without bound until the machine OOMs (pinned pages can't
swap). This crashed a 1B EBFT run and a 128B unpacked-SFT run. Fix: the free
pool is an LRU (OrderedDict, MRU last) capped by total free bytes; returning a
buffer over the cap evicts least-recently-used shape buckets first. The cap is
``max(max_pool_bytes, 1.5 x largest single pass)`` — the working-set floor lets
a legitimately large *shape-stable* run keep its whole pass pooled (no thrash),
while the cross-pass leak (many passes' worth of stale shapes) is evicted down
to ~1.5 passes. ``EXL3_OFFLOAD_POOL_GIB`` overrides the floor.

Value-exact: copies don't round. A same-seed run must produce bit-identical
losses to sync offload (and to no offload) — that is the verification gate.

Limitations (both raise clearly rather than corrupt):
  * one forward pass per context entry, no nesting;
  * a record can be unpacked once (no ``retain_graph``/double backward —
    the trainers never do that; use offload_mode=sync if one ever does).
"""

from __future__ import annotations
from typing import Optional
import collections
import os
import torch

# Free-pool ceiling below which a shape-stable run is never touched. The
# effective cap is raised to ~1.5x the largest observed single pass so large
# stable runs don't thrash; this only bounds the cross-pass leak. Override with
# EXL3_OFFLOAD_POOL_GIB.
_DEFAULT_POOL_BYTES = 4 * (1 << 30)  # 4 GiB


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class _Record:
    __slots__ = ("cpu", "gpu", "device", "dtype", "event", "consumed")

    def __init__(self, cpu, device, dtype, event):
        self.cpu = cpu            # pinned host buffer (owned until consumed)
        self.gpu = None           # device tensor once prefetched/unpacked
        self.device = device
        self.dtype = dtype
        self.event = event        # last side-stream op touching this record
        self.consumed = False


class AsyncActivationOffload:
    """Reusable context manager; hold ONE per net and re-enter every forward.

    Enter wraps the decoder block loop the same way ``save_on_cpu`` did;
    unpacks happen later, during backward, after exit. The pool, streams and
    in-flight records all live on this object, which is why it must persist
    across steps rather than being rebuilt per forward.
    """

    def __init__(self, min_bytes: int = 1 << 20, prefetch: int = 1,
                 max_pool_bytes: Optional[int] = None):
        self.min_bytes = int(min_bytes)
        self.prefetch = max(0, int(prefetch))
        if max_pool_bytes is None:
            gib = os.environ.get("EXL3_OFFLOAD_POOL_GIB")
            max_pool_bytes = float(gib) * (1 << 30) if gib else _DEFAULT_POOL_BYTES
        self.max_pool_bytes = int(max_pool_bytes)
        self._streams: dict = {}                       # device index -> Stream
        # (shape, dtype) -> [(cpu, event), ...], ordered LRU (most-recent last).
        self._pool: "collections.OrderedDict" = collections.OrderedDict()
        self._pool_bytes = 0                           # total free bytes held in the pool
        self._working_set = 0                          # largest single-pass offloaded bytes seen
        self._pass_bytes = 0                           # offloaded bytes in the currently open pass
        self._pass_records: Optional[list] = None      # records of the open pass
        self._hooks_ctx = None

    # --- plumbing -----------------------------------------------------------

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self._streams.get(device.index)
        if s is None:
            s = torch.cuda.Stream(device=device)
            self._streams[device.index] = s
        return s

    def _get_pinned(self, shape, dtype) -> torch.Tensor:
        """A pinned host buffer, reused once its previous HtoD readback is done."""
        key = (shape, dtype)
        free = self._pool.get(key)
        while free:
            cpu, event = free.pop()
            self._pool_bytes -= _nbytes(cpu)
            if event is None or event.query():
                if free:
                    self._pool.move_to_end(key)        # bucket still live → mark MRU
                else:
                    del self._pool[key]
                return cpu
            # Still in flight (backward hasn't caught up); don't block on it —
            # buffers come back in roughly FIFO order, so the rest of the list
            # is no more likely to be ready. Fall through to a fresh alloc.
            free.append((cpu, event))
            self._pool_bytes += _nbytes(cpu)
            self._pool.move_to_end(key)
            break
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def _return_pinned(self, cpu: torch.Tensor, event) -> None:
        """Return a consumed pinned buffer to the pool, then enforce the cap."""
        key = (tuple(cpu.shape), cpu.dtype)
        bucket = self._pool.get(key)
        if bucket is None:
            bucket = self._pool[key] = []
        bucket.append((cpu, event))
        self._pool.move_to_end(key)                    # just used → MRU
        self._pool_bytes += _nbytes(cpu)
        # Effective cap floors at ~1.5x the largest single pass so a large but
        # shape-stable run keeps its whole working set pooled (no re-alloc
        # thrash); only the cross-pass leak of stale shapes is evicted.
        cap = max(self.max_pool_bytes, int(self._working_set * 3 // 2))
        while self._pool_bytes > cap and self._pool:
            lru_key = next(iter(self._pool))           # least-recently-used bucket
            lru = self._pool[lru_key]
            evict_cpu, evict_event = lru.pop(0)
            # The event is the last HtoD readback that read this buffer; the
            # pinned memory can't be released until that copy has finished.
            if evict_event is not None and not evict_event.query():
                evict_event.synchronize()
            self._pool_bytes -= _nbytes(evict_cpu)
            if not lru:
                del self._pool[lru_key]

    # --- saved_tensors_hooks ------------------------------------------------

    def _pack(self, t: torch.Tensor):
        if (not t.is_cuda) or t.numel() * t.element_size() < self.min_bytes:
            return ("keep", t)
        records = self._pass_records
        self._pass_bytes += _nbytes(t)
        stream = self._stream(t.device)
        cpu = self._get_pinned(tuple(t.shape), t.dtype)
        stream.wait_stream(torch.cuda.current_stream(t.device))
        with torch.cuda.stream(stream):
            cpu.copy_(t, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        # The block's activation memory may be freed (that's the point) and
        # reused by compute-stream work; the side stream is still reading it.
        t.record_stream(stream)
        rec = _Record(cpu, t.device, t.dtype, event)
        records.append(rec)
        return ("cpu", records, len(records) - 1)

    def _start_htod(self, rec: _Record):
        if rec.gpu is not None or rec.consumed:
            return
        stream = self._stream(rec.device)
        # rec.gpu is allocated on the COMPUTE stream: the caching allocator may
        # back it with memory whose previous owner still has pending compute
        # work (reuse is only implicitly ordered within one stream). The side
        # stream writes it, so it must first wait for the compute stream as of
        # now -- without this the copy races whatever last used that memory
        # (bit-garbage/NaN grads, found the hard way).
        rec.gpu = torch.empty(rec.cpu.shape, dtype=rec.dtype, device=rec.device)
        stream.wait_stream(torch.cuda.current_stream(rec.device))
        with torch.cuda.stream(stream):
            # Same stream as the DtoH, so "copy back" is ordered after
            # "copy out" by stream order alone.
            rec.gpu.copy_(rec.cpu, non_blocking=True)
            rec.event = torch.cuda.Event()
            rec.event.record(stream)
        # Written on the side stream, later read on the compute stream (which
        # wait_events first); keep the allocator honest about the writer.
        rec.gpu.record_stream(stream)

    def _unpack(self, packed):
        tag = packed[0]
        if tag == "keep":
            return packed[1]
        _, records, idx = packed
        rec = records[idx]
        if rec.consumed:
            raise RuntimeError(
                "async activation offload: saved tensor unpacked twice "
                "(retain_graph / double backward?). Use offload_mode=sync.")
        self._start_htod(rec)
        # Prefetch the records backward will want next (reverse pack order).
        started = 0
        for j in range(idx - 1, -1, -1):
            if started >= self.prefetch:
                break
            nxt = records[j]
            if nxt.gpu is None and not nxt.consumed:
                self._start_htod(nxt)
                started += 1
        torch.cuda.current_stream(rec.device).wait_event(rec.event)
        gpu, cpu = rec.gpu, rec.cpu
        rec.consumed = True
        rec.gpu = rec.cpu = None
        # The host buffer is reusable once the HtoD has actually read it.
        self._return_pinned(cpu, rec.event)
        return gpu

    # --- context ------------------------------------------------------------

    def __enter__(self):
        assert self._hooks_ctx is None, \
            "AsyncActivationOffload does not support nested/concurrent passes"
        self._pass_records = []
        self._pass_bytes = 0
        self._hooks_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack, self._unpack)
        self._hooks_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        ctx, self._hooks_ctx = self._hooks_ctx, None
        # The records list stays alive via the pack handles inside the autograd
        # graph; backward consumes it after this exit.
        self._pass_records = None
        # Track the largest pass so the cap can floor at the real working set.
        if self._pass_bytes > self._working_set:
            self._working_set = self._pass_bytes
        return ctx.__exit__(*exc)
