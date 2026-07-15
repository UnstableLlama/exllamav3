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
import torch


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

    def __init__(self, min_bytes: int = 1 << 20, prefetch: int = 1):
        self.min_bytes = int(min_bytes)
        self.prefetch = max(0, int(prefetch))
        self._streams: dict = {}                       # device index -> Stream
        self._pool = collections.defaultdict(list)     # (shape, dtype) -> [(cpu, event)]
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
        free = self._pool.get((shape, dtype))
        while free:
            cpu, event = free.pop()
            if event is None or event.query():
                return cpu
            # Still in flight (backward hasn't caught up); don't block on it —
            # buffers come back in roughly FIFO order, so the rest of the list
            # is no more likely to be ready. Fall through to a fresh alloc.
            free.append((cpu, event))
            break
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    # --- saved_tensors_hooks ------------------------------------------------

    def _pack(self, t: torch.Tensor):
        if (not t.is_cuda) or t.numel() * t.element_size() < self.min_bytes:
            return ("keep", t)
        records = self._pass_records
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
        self._pool[(tuple(cpu.shape), rec.dtype)].append((cpu, rec.event))
        return gpu

    # --- context ------------------------------------------------------------

    def __enter__(self):
        assert self._hooks_ctx is None, \
            "AsyncActivationOffload does not support nested/concurrent passes"
        self._pass_records = []
        self._hooks_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack, self._unpack)
        self._hooks_ctx.__enter__()
        return self

    def __exit__(self, *exc):
        ctx, self._hooks_ctx = self._hooks_ctx, None
        # The records list stays alive via the pack handles inside the autograd
        # graph; backward consumes it after this exit.
        self._pass_records = None
        return ctx.__exit__(*exc)
