"""
Does the v1.0.0+ fused trellis GEMM beat the training path at TRAINING shapes?

Context (see doc/qlora_handoff.md Session 48, doc/qlora_optimization_audit.md
A1): audit A1's original plan -- call the fused trellis GEMM for the frozen-
base forward matmul instead of reconstructing -- was dropped in S30 because
the fused kernel was decode-only (inference dispatches it at rows <=
AUTO_RECONSTRUCT_THRESHOLD = 144 and reconstructs at prefill shapes).
Upstream v1.0.0 "greatly improved GEMM/GEMV performance on Ampere"; this
script measures whether that reopened the door. science/qgemm_benchmark.py
only sweeps decode shapes (m = 1/4/16); here m spans training shapes
(batch x seq flattened rows).

Arms, per quantized weight and per m (all producing y = x @ W_eff):

  fused[pref]  ext.exl3_gemm, its own preferred kernel (suh/had/svh fused in)
  fused[best]  best over all shape-compatible kernel indices (sweep)
  recon_hgemm  inference's own prefill path: had_r_128 + fp16 reconstruct +
               ext.hgemm + had_r_128
  train_bf16   what training pays TODAY per base-matmul call (S36 fast path):
               per-call bf16 inner reconstruct + EXL3LoRAHadFunction.
               _base_matmul (torch blockwise-Hadamard sandwich)
  train_hadk   train_bf16's sandwich swapped for the ext.had_r_128 kernels
               (fp16) -- probes the S35 88 ms/step "had_transform" bucket
               separately from the fused-GEMM question
  dense_bf16   pre-built W_eff, torch matmul only -- the residency ceiling
               (no reconstruct, no Hadamard; 2 bytes/param resident)

Reading the results:
  * fused[best] < train_bf16  =>  audit A1's original fix is BACK ON: use the
    fused GEMM in forward + checkpoint-recompute (the base is a constant --
    autograd doesn't care), reconstruct only in backward for grad_y @ W^T.
    Per linear per step: reconstructs 3 -> 1 and the forward Hadamard
    sandwich disappears (S35 buckets: ~2/3 of 88 ms had + ~2/3 of 29 ms
    reconstruct + whatever the base-GEMM delta is, on the 3B profile config).
  * fused loses but train_hadk < train_bf16  =>  keep the reconstruct
    strategy, wire had_r_128 into the training sandwich instead (forward
    uses the kernel; the backward adjoint is the same kernel with suh/svh
    swapped -- H^T = H).
  * both lose  =>  the door stays closed; the S35 ladder stands as-is.

Run (box, one GPU is enough):
  python science/qgemm_train_shapes.py --model /path/to/exl3-model --device cuda:1
  python science/qgemm_train_shapes.py --bpw 2,4,6 --kn 3072x3072,3072x8192 \\
      --device cuda:1          # synthetic weights (timing-faithful, no model)

Correctness is gated either way: every arm's output is checked against a full
fp32 reference on the first buffer (max rel err printed; fused/fp16 arms are
expected in the ~1e-3 half-precision band, bf16 arms ~1e-2).
"""

import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.memory import free_mem
from exllamav3.training import backbone
from exllamav3.training.qlora_linear import EXL3LoRAHadFunction
from tabulate import tabulate

torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

MAX_RECONSTRUCT_SLICE_N = 32768   # mirrors modules/quant/exl3.py


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None,
                    help="EXL3 model dir; unique linear shapes are benchmarked "
                         "on the REAL trellis tensors. Omit for synthetic.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", default="16,256,1024,2048,4096,8192",
                    help="Comma list of row counts (batch x seq flattened). "
                         "16 is a decode-regime sanity row.")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--max-shapes", type=int, default=6,
                    help="Cap distinct weight shapes benchmarked (model mode).")
    # Synthetic-mode knobs (ignored with --model):
    ap.add_argument("--bpw", default="4", help="Comma list of K (bits per weight).")
    ap.add_argument("--kn", default="3072x3072,3072x8192,8192x3072",
                    help="Comma list of KxN shapes, e.g. 4096x4096.")
    ap.add_argument("--mcg", action="store_true", help="Synthetic: mcg codebook flag.")
    ap.add_argument("--mul1", action="store_true", help="Synthetic: mul1 codebook flag.")
    return ap.parse_args()


def collect_model_weights(model_dir, device, max_shapes):
    """(label, trellis, suh, svh, K, mcg, mul1) for each distinct (k, n, K)."""
    from exllamav3 import Config, Model
    from exllamav3.modules.quant.exl3 import LinearEXL3
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(device, progressbar=True)
    seen, out = set(), []
    stack = list(model.modules)
    while stack:
        mod = stack.pop(0)
        stack.extend(getattr(mod, "modules", []) or [])
        inner = getattr(mod, "inner", None)
        if not isinstance(inner, LinearEXL3):
            continue
        sig = (inner.in_features, inner.out_features, inner.K)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((f"{getattr(mod, 'key', 'linear')} "
                    f"[{inner.in_features}x{inner.out_features} K={inner.K}]",
                    inner.trellis, inner.suh, inner.svh,
                    inner.K, inner.mcg, inner.mul1))
        if len(out) >= max_shapes:
            break
    assert out, "no LinearEXL3 modules found in the model"
    return out


def synthetic_weights(kns, ks_bpw, device, mcg, mul1):
    """Random trellis data: garbage weights, timing- and parity-faithful
    (any int16 payload decodes to a valid codebook word)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    out = []
    for kn in kns:
        k, n = (int(v) for v in kn.split("x"))
        assert k % 128 == 0 and n % 128 == 0, "shapes must be 128-aligned"
        for K in ks_bpw:
            trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K),
                                    generator=g, dtype=torch.int16).to(device)
            suh = (torch.randint(0, 2, (k,), generator=g, dtype=torch.int8)
                   .to(device).to(torch.half) * 2.0 - 1.0)
            svh = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int8)
                   .to(device).to(torch.half) * 2.0 - 1.0)
            out.append((f"synthetic [{k}x{n} K={K}]",
                        trellis, suh, svh, K, mcg, mul1))
    return out


def bench(fn, runs, device, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(runs):
        fn()
    ev1.record()
    ev1.synchronize()
    return ev0.elapsed_time(ev1) / runs


def rel_err(y, y_ref32):
    return ((y.float() - y_ref32).abs().max() / y_ref32.abs().max()).item()


def main():
    args = parse_args()
    device = args.device
    ms = [int(v) for v in args.m.split(",")]
    n_kernels = ext.exl3_gemm_num_kernel_shapes()

    if args.model:
        weights = collect_model_weights(args.model, device, args.max_shapes)
    else:
        weights = synthetic_weights(args.kn.split(","),
                                    [int(v) for v in args.bpw.split(",")],
                                    device, args.mcg, args.mul1)

    headers = ["shape", "m", "fused[pref]", "fused[best]", "recon_hgemm",
               "train_bf16", "train_hadk", "dense_bf16", "best/train"]
    rows, err_rows = [], []

    for label, trellis, suh, svh, K, mcg, mul1 in weights:
        k = suh.shape[0]
        n = svh.shape[0]
        if n > MAX_RECONSTRUCT_SLICE_N:
            print(f"-- skipping {label}: n > MAX_RECONSTRUCT_SLICE_N "
                  f"(vocab-head class; benchmark the body linears)")
            continue
        free_mem()

        # fp32 reference weight W_eff = diag(suh) . H . W_inner . H . diag(svh),
        # realized by pushing the identity through the training sandwich.
        w16 = torch.empty((k, n), dtype=torch.half, device=device)
        ext.reconstruct(w16, trellis, K, mcg, mul1)
        had32 = backbone.hadamard_128(torch.device(device), torch.float32)
        w_eff32 = EXL3LoRAHadFunction._base_matmul(
            torch.eye(k, dtype=torch.float32, device=device), w16.float(),
            suh.float(), svh.float(), had32)
        w_eff_bf16 = w_eff32.to(torch.bfloat16)

        had_bf16 = backbone.hadamard_128(torch.device(device), torch.bfloat16)
        suh_bf16, svh_bf16 = suh.to(torch.bfloat16), svh.to(torch.bfloat16)

        for m in ms:
            # A few input buffers to cycle so the L2 doesn't flatter small m.
            n_buf = 4
            xs16 = [torch.randn(m, k, dtype=torch.half, device=device) * 0.5
                    for _ in range(n_buf)]
            xs_bf = [x.to(torch.bfloat16) for x in xs16]
            y16 = torch.empty(m, n, dtype=torch.half, device=device)
            xh = torch.empty(m, k, dtype=torch.half, device=device)
            step = {"i": 0}

            def nx():
                step["i"] = (step["i"] + 1) % n_buf
                return step["i"]

            # -- fused trellis GEMM (preferred kernel + sweep) ----------------
            def fused(idx):
                ext.exl3_gemm(xs16[nx()], trellis, y16, suh, xh, svh,
                              idx, mcg, mul1, 0)

            fused(-1)
            e_fused = rel_err(y16, xs16[step["i"]].float() @ w_eff32)
            t_pref = bench(lambda: fused(-1), args.runs, device)
            t_best, best_idx = t_pref, "pref"
            for idx in range(1, n_kernels + 1):
                if not ext.exl3_gemm_shape_compat(idx, m, k, n, K):
                    continue
                t = bench(lambda: fused(idx), args.runs, device)
                if t < t_best:
                    t_best, best_idx = t, idx

            # -- inference prefill path (reconstruct + hgemm, had kernels) ----
            def recon_hgemm():
                x = xs16[nx()]
                ext.had_r_128(x, xh, suh, None, 1.0)
                w = torch.empty((k, n), dtype=torch.half, device=device)
                ext.reconstruct(w, trellis, K, mcg, mul1)
                ext.hgemm(xh, w, y16)
                ext.had_r_128(y16, y16, None, svh, 1.0)

            recon_hgemm()
            e_recon = rel_err(y16, xs16[step["i"]].float() @ w_eff32)
            t_recon = bench(recon_hgemm, args.runs, device)

            # -- training fast path (S36): bf16 reconstruct + torch sandwich --
            def train_bf16():
                w = torch.empty((k, n), dtype=torch.bfloat16, device=device)
                ext.reconstruct(w, trellis, K, mcg, mul1)
                return EXL3LoRAHadFunction._base_matmul(
                    xs_bf[nx()], w, suh_bf16, svh_bf16, had_bf16)

            y = train_bf16()
            e_train = rel_err(y, xs_bf[step["i"]].float() @ w_eff32)
            t_train = bench(train_bf16, args.runs, device)

            # -- training sandwich with the had_r_128 kernels (fp16 probe) ----
            def train_hadk():
                x = xs16[nx()]
                ext.had_r_128(x, xh, suh, None, 1.0)
                w = torch.empty((k, n), dtype=torch.half, device=device)
                ext.reconstruct(w, trellis, K, mcg, mul1)
                y = xh @ w
                ext.had_r_128(y, y, None, svh, 1.0)
                return y

            y = train_hadk()
            e_hadk = rel_err(y, xs16[step["i"]].float() @ w_eff32)
            t_hadk = bench(train_hadk, args.runs, device)

            # -- residency ceiling: dense bf16 matmul, weight pre-built -------
            def dense():
                return xs_bf[nx()] @ w_eff_bf16

            y = dense()
            e_dense = rel_err(y, xs_bf[step["i"]].float() @ w_eff32)
            t_dense = bench(dense, args.runs, device)

            rows.append([label, m,
                         f"{t_pref:.3f}",
                         f"{t_best:.3f} [{best_idx}]",
                         f"{t_recon:.3f}", f"{t_train:.3f}",
                         f"{t_hadk:.3f}", f"{t_dense:.3f}",
                         f"{t_train / t_best:.2f}x"])
            if m == ms[0]:
                err_rows.append([label, f"{e_fused:.2e}", f"{e_recon:.2e}",
                                 f"{e_train:.2e}", f"{e_hadk:.2e}",
                                 f"{e_dense:.2e}"])
            del xs16, xs_bf, y16, xh
            free_mem()

    print()
    print(f"device {device}, {args.runs} runs/cell, ms/call "
          f"(best/train > 1.00x means the fused GEMM WINS at that shape)")
    print()
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()
    print("max rel err vs fp32 reference (first m only; fp16 arms ~1e-3, "
          "bf16 arms ~1e-2 expected):")
    print(tabulate(err_rows, headers=["shape", "fused", "recon_hgemm",
                                      "train_bf16", "train_hadk", "dense_bf16"],
                   tablefmt="github"))


if __name__ == "__main__":
    main()
