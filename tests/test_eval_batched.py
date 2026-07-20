"""
CPU tests for batched held-out evaluation (``--eval-batch``, audit A6):
``NativeLlamaQLoRA.compute_loss_per_seq``.

The eval metric is defined as the mean of each example's own token-mean loss
(the batch-1 ``eval_loss`` loop, matched with the BNB arm). Batched eval must
reproduce that definition exactly: these tests check, on a mock net whose
forward is a per-position embedding lookup (so padding cannot leak into real
rows -- the property real causal attention provides), that

  * per-row ``sums / counts`` from a right-padded batch equals per-example
    ``compute_loss`` run one at a time, on BOTH fused heads (single-shot and
    vocab-chunked), with and without the Gemma softcap;
  * the trainable-head path (``train_head``) agrees the same way;
  * an all-masked row contributes an exact 0 with count 0 (matching the
    batch-1 zero-supervision behavior).

No GPU / compiled extension / real model needed. Run:
    python tests/test_eval_batched.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_qll = _load("qlora_linear")
_fce = _load("fused_ce")
_nl = _load("native_llama")

NativeLlamaQLoRA = _nl.NativeLlamaQLoRA
IGNORE_INDEX = _fce.IGNORE_INDEX

V, D = 40, 16
PAD_ID = 0


def _mock_net(weight, *, vocab_chunk=0, softcap=0.0, train_head=False):
    """A headless net exposing exactly what compute_loss/compute_loss_per_seq
    touch. The forward is a per-position embedding lookup: padded positions
    can't perturb real ones, which is the property the causal forward + mask
    guarantees (exercised on the real path by test_native_llama's packing/pad
    tests) -- here it isolates the head/reduction math under test."""
    net = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    torch.nn.Module.__init__(net)   # bare registries so Parameters can attach
    emb = torch.randn(V, D, dtype=torch.float64) * 0.7

    def fwd(input_ids, attention_mask=None, position_ids=None, seg_ids=None):
        return emb[input_ids]

    net.forward = fwd
    net.train_head = train_head
    net.lora_head = False
    net._adapters_off = False
    net._head_device = torch.device("cpu")
    net.final_softcap = softcap
    net.head_vocab_chunk = vocab_chunk
    if vocab_chunk > 0:
        net._head_slice = (lambda s, n: weight[:, s:s + n], V, 1)
    else:
        net._head_slice = None
    net.lm_head_weight_fn = lambda: (lambda: weight)
    if train_head:
        net.head_weight = weight
    return net


def _examples(seed):
    """Ragged examples incl. prompt masking and one zero-supervision row."""
    g = torch.Generator().manual_seed(seed)
    exs = []
    for length, mask_to in ((11, 4), (7, 2), (9, 9), (5, 1), (12, 6)):
        ids = torch.randint(1, V, (length,), generator=g).tolist()
        labels = list(ids)
        for i in range(min(mask_to, length)):
            labels[i] = IGNORE_INDEX
        exs.append({"input_ids": ids, "labels": labels})
    return exs


def _collate(batch):
    """Right-pad like the trainer's collate (labels -100, ids PAD_ID)."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels = [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [PAD_ID] * pad)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad)
    return torch.tensor(ids), torch.tensor(labels)


def _check_parity(net, exs, tag, atol=1e-5):
    with torch.no_grad():
        singles = []
        for ex in exs:
            ids, labels = _collate([ex])
            singles.append(net.compute_loss(ids, labels, chunk=3).item())
        ids, labels = _collate(exs)
        sums, counts = net.compute_loss_per_seq(ids, labels, chunk=3)
    means = (sums / counts.clamp(min=1)).tolist()
    for i, (a, b) in enumerate(zip(means, singles)):
        assert abs(a - b) < atol, f"{tag}: row {i} batched {a} vs batch-1 {b}"
    # The all-masked row is an exact zero with count 0, like batch-1.
    zero_rows = [i for i, ex in enumerate(exs)
                 if all(l == IGNORE_INDEX for l in ex["labels"])]
    for i in zero_rows:
        assert counts[i].item() == 0 and sums[i].item() == 0.0, \
            f"{tag}: masked row {i} not (0, 0)"
    print(f"[eval-batch] {tag} parity PASSED")


def test_fused_head_parity():
    torch.manual_seed(0)
    w = torch.randn(D, V, dtype=torch.float64)
    _check_parity(_mock_net(w), _examples(1), "fused single-shot")


def test_vocab_chunked_parity():
    torch.manual_seed(1)
    w = torch.randn(D, V, dtype=torch.float64)
    _check_parity(_mock_net(w, vocab_chunk=16), _examples(2), "vocab-chunked")


def test_softcap_parity():
    torch.manual_seed(2)
    w = torch.randn(D, V, dtype=torch.float64) * 3
    _check_parity(_mock_net(w, softcap=10.0), _examples(3), "softcap fused")
    _check_parity(_mock_net(w, vocab_chunk=8, softcap=10.0), _examples(4),
                  "softcap vocab-chunked")


def test_trainable_head_parity():
    torch.manual_seed(3)
    w = torch.nn.Parameter(torch.randn(D, V, dtype=torch.float64))
    _check_parity(_mock_net(w, train_head=True), _examples(5), "trainable head")


def test_zero_supervision_batch():
    """A whole batch of masked rows: sums/counts all zero, no NaN."""
    torch.manual_seed(4)
    w = torch.randn(D, V, dtype=torch.float64)
    net = _mock_net(w)
    exs = _examples(6)
    for ex in exs:
        ex["labels"] = [IGNORE_INDEX] * len(ex["labels"])
    ids, labels = _collate(exs)
    with torch.no_grad():
        sums, counts = net.compute_loss_per_seq(ids, labels, chunk=3)
    assert counts.sum().item() == 0 and sums.abs().sum().item() == 0.0
    means = sums / counts.clamp(min=1)
    assert torch.isfinite(means).all()
    print("[eval-batch] zero-supervision batch PASSED")


def main():
    test_fused_head_parity()
    test_vocab_chunked_parity()
    test_softcap_parity()
    test_trainable_head_parity()
    test_zero_supervision_batch()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
