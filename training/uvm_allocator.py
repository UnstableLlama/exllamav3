"""
Opt-in VRAM spillover on Linux via CUDA unified memory (a pluggable allocator).

Why this exists: on Windows, the NVIDIA WDDM driver (>= 536.40) has a "Sysmem
Fallback Policy" (NVIDIA Control Panel -> Manage 3D Settings) that transparently
spills CUDA allocations into system RAM ("Shared GPU memory") when the GPU
fills up, so a slightly-too-big run slows down instead of dying. The Linux
driver has no such fallback: cudaMalloc simply fails and PyTorch raises
torch.cuda.OutOfMemoryError the moment the working set exceeds VRAM. That
asymmetry is why the same run can "just work" (slowly) on a Windows box and
insta-OOM on Linux.

This module recreates the Windows behavior on Linux by swapping PyTorch's
caching allocator for a tiny cudaMallocManaged one (CUDA unified memory, UVM):
every tensor is allocated as managed memory whose preferred location is the
GPU, and once the GPU is oversubscribed the driver evicts cold pages to host
RAM on demand (Pascal or newer; oversubscription is a Linux-only UVM feature,
which is why this path isn't useful on Windows). The allocator .so is compiled
once with nvcc on first use and cached under ~/.cache/exl3_qlora_uvm/.

Costs and sharp edges -- this is a lever for runs that fit MOSTLY in VRAM:

 - A few percent of spillover is usable (cold pages live in host RAM and are
   read over PCIe); meaningful oversubscription (say 1.5-2x VRAM) crawls.
   Spilling to host RAM is ~10-50x slower than HBM/GDDR per access.
 - There is no caching layer: every tensor alloc/free is a real CUDA call, so
   allocation-heavy code runs somewhat slower even before anything spills.
 - torch.cuda memory stats (max_memory_allocated, memory_stats,
   reset_peak_memory_stats) are NOT supported by pluggable allocators and
   raise; callers must guard (qlora_train_native.py reports peak VRAM as n/a).
   torch.cuda.empty_cache() and set_per_process_memory_fraction() become
   no-ops.
 - Must be installed BEFORE the first CUDA tensor exists in the process --
   PyTorch refuses to swap an allocator that has already served an allocation.
 - Allocations no longer fail at VRAM capacity, only when host RAM is also
   exhausted (at which point the Linux OOM killer, not a catchable torch OOM,
   is the likely failure mode). Autosplit heuristics that probe device
   capacity therefore stop making sense: --parallel split rejects this flag.

Usage: qlora_train_native.py --vram-spillover, or vram_spillover: true in the
YAML config (parallel: single only).
"""

import hashlib
import os
import shutil
import subprocess

import torch

# The two entry points PyTorch's CUDAPluggableAllocator dlopens. Signatures per
# the torch.cuda.memory.CUDAPluggableAllocator docs. On alloc failure the CUDA
# error is cleared and nullptr returned (torch has no OOM-retry path to feed
# here; with UVM a failure means host RAM is exhausted too, so the run is dead
# either way -- the fprintf is so the cause is visible in the log).
_UVM_SRC = r"""
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cstdio>

extern "C" {

void* uvm_alloc(ssize_t size, int device, cudaStream_t stream) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, " !! uvm_alloc: cudaMallocManaged(%zd bytes) failed: %s\n",
                (ssize_t) size, cudaGetErrorString(err));
        cudaGetLastError();
        return nullptr;
    }
    // Keep pages resident on the GPU while they fit; under oversubscription
    // the driver evicts cold pages to host RAM. AccessedBy keeps the GPU
    // mapping alive so evicted pages are read over PCIe instead of faulting
    // pages back and forth (the thrash mode).
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
    return ptr;
}

void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (ptr != nullptr)
        cudaFree(ptr);
}

}
"""

_enabled = False


def is_enabled() -> bool:
    """True once the UVM allocator has been installed in this process."""
    return _enabled


def _find_nvcc() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME:
        cand = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.isfile(cand):
            return cand
    return shutil.which("nvcc")


def _build_allocator_so(verbose: bool) -> str:
    """Compile the allocator .so (once; content-hashed cache) and return its path."""
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "exl3_qlora_uvm",
    )
    os.makedirs(cache_dir, exist_ok=True)
    tag = hashlib.sha256(_UVM_SRC.encode()).hexdigest()[:16]
    so_path = os.path.join(cache_dir, f"uvm_allocator_{tag}.so")
    if os.path.isfile(so_path):
        return so_path

    nvcc = _find_nvcc()
    if nvcc is None:
        raise SystemExit(
            "--vram-spillover needs nvcc to build its allocator (the same CUDA "
            "toolkit that built the exllamav3 extension). Set CUDA_HOME or put "
            "nvcc on PATH.")
    src_path = so_path[:-3] + ".cu"
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(_UVM_SRC)
    if verbose:
        print(f" -- building UVM allocator (one-time): {so_path}")
    tmp_path = so_path + f".tmp{os.getpid()}"
    proc = subprocess.run(
        [nvcc, "-O2", "--shared", "-Xcompiler", "-fPIC", src_path, "-o", tmp_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"--vram-spillover: nvcc failed to build the UVM allocator:\n"
            f"{proc.stdout}\n{proc.stderr}")
    os.replace(tmp_path, so_path)  # atomic, so concurrent runs never dlopen a partial .so
    return so_path


def enable(verbose: bool = True) -> None:
    """Swap PyTorch's CUDA allocator for the managed-memory one.

    Call before ANY CUDA tensor is created (i.e. right after arg parsing,
    before model load). Idempotent within a process.
    """
    global _enabled
    if _enabled:
        return
    if os.name == "nt":
        raise SystemExit(
            "--vram-spillover is the Linux substitute for a feature Windows "
            "already has in the driver: enable 'CUDA - Sysmem Fallback Policy' "
            "in the NVIDIA Control Panel instead (UVM oversubscription, which "
            "this flag relies on, is not supported on Windows).")
    if not torch.cuda.is_available():
        raise SystemExit("--vram-spillover requires CUDA.")
    # The native caching allocator's env config doesn't apply to a pluggable
    # allocator; expandable_segments etc. are silently ignored, but an explicit
    # backend swap is a config conflict worth failing loudly on.
    if "backend:cudaMallocAsync" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise SystemExit(
            "--vram-spillover conflicts with PYTORCH_CUDA_ALLOC_CONF="
            "backend:cudaMallocAsync -- unset the backend override first.")
    if torch.cuda.is_initialized():
        raise SystemExit(
            "--vram-spillover must be enabled before the first CUDA tensor is "
            "created; something initialized CUDA earlier in this process.")

    so_path = _build_allocator_so(verbose)
    allocator = torch.cuda.memory.CUDAPluggableAllocator(so_path, "uvm_alloc", "uvm_free")
    torch.cuda.memory.change_current_allocator(allocator)
    _enabled = True
    if verbose:
        print(" -- VRAM spillover enabled: CUDA unified memory allocator "
              "(cudaMallocManaged). Runs that exceed VRAM spill cold pages to "
              "host RAM instead of OOMing; expect a slowdown proportional to "
              "the spill. Peak-VRAM stats are unavailable in this mode.")
