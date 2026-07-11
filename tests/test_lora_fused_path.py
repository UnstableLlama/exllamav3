"""
Guards for the fused-decode / runtime-LoRA interaction (Session 22 finding).

The fused multi-linear decode kernels (exl3_mgemm over MultiLinear pairs, the
BC_GatedMLP bsz-1 graph) read trellis storage directly and never call
Linear.forward -- which is the only place a runtime LoRA (model.lora.LoRA) is
applied. Before the fix, token-by-token generation on an EXL3-quantized base
silently dropped the k/v (and gated-attn q), gate/up (and BC-path down) LoRA
components, so an adapter that visibly steered the training-side forward
looked like a no-op at inference. Attention (attn/sliding_attn) and GatedMLP
now fall back to the per-linear path while any involved Linear carries LoRA
tensors.

Because the failure mode is SILENT (generation stays coherent, the adapter
just doesn't run), these tests are tripwires on the guard conditions in the
source itself, dependency-free so they run in any container. The semantics of
the helper are tested with real imports where torch is available. The real
end-to-end check -- an adapter visibly changing bsz=1 decode output on an EXL3
quant, fused kernels present -- needs a GPU and a quantized model; it is on
the handoff box list, not here.
"""

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*rel: str) -> str:
    with open(os.path.join(REPO, *rel), encoding = "utf8") as f:
        return f.read()


def test_helper_defined():
    src = _src("exllamav3", "modules", "linear.py")
    assert "def has_runtime_lora(" in src


@pytest.mark.parametrize("fname", ["attn.py", "sliding_attn.py"])
def test_attention_fused_branches_guarded(fname):
    src = _src("exllamav3", "modules", fname)
    # q/g fused branch falls back when q_proj or g_proj carries a LoRA
    assert re.search(
        r"self\.multi_qg is None or bsz \* q_len > 32[^:]*?"
        r"has_runtime_lora\(self\.q_proj, self\.g_proj\)",
        src, re.S), f"{fname}: multi_qg branch lost its runtime-LoRA guard"
    # k/v fused branch falls back when k_proj or v_proj carries a LoRA
    assert re.search(
        r"self\.multi_kv is None or bsz \* q_len > 32[^:]*?"
        r"has_runtime_lora\(self\.k_proj, self\.v_proj\)",
        src, re.S), f"{fname}: multi_kv branch lost its runtime-LoRA guard"


def test_gated_mlp_fused_branches_guarded():
    src = _src("exllamav3", "modules", "mlp.py")
    # Checked across ALL slices (uniform branch choice), once per forward.
    assert "gu_lora = has_runtime_lora(*self.gates, *self.ups)" in src
    assert "down_lora = has_runtime_lora(*self.downs)" in src
    # gate/up mgemm branch falls back when gate or up carries a LoRA
    assert re.search(
        r"self\.multi_gu\[s\] is None or bsz \* q_len > 32 or gu_lora",
        src), "GatedMLP: multi_gu branch lost its runtime-LoRA guard"
    # the fully-fused BC bsz-1 branch also skips down_proj -> must yield to the
    # mgemm branch (which calls downs[s].forward) when down carries a LoRA
    assert re.search(
        r"self\.bc is not None and bsz == 1 and q_len == 1 and not down_lora",
        src), "GatedMLP: BC bsz-1 branch lost its runtime-LoRA guard"


def test_moe_expert_lora_warning_present():
    # Routed experts (BlockSparseMLP) have no per-linear fallback yet; the
    # loader must warn instead of silently no-opping the adapter.
    src = _src("exllamav3", "model", "lora.py")
    assert re.search(r"\\\.experts\\\.", src) and "not applied" in src, \
        "lora.py: routed-expert no-op warning removed without a MoE fallback"


def test_has_runtime_lora_semantics():
    torch = pytest.importorskip("torch")  # noqa: F841  (linear.py imports it)
    try:
        from exllamav3.modules.linear import has_runtime_lora
    except ImportError as e:
        pytest.skip(f"exllamav3 not importable here: {e}")

    class Stub:
        def __init__(self, loaded: bool):
            self.lora_a_tensors = {object(): object()} if loaded else {}

    assert not has_runtime_lora()
    assert not has_runtime_lora(None)
    assert not has_runtime_lora(Stub(False), None, Stub(False))
    assert has_runtime_lora(Stub(True))
    assert has_runtime_lora(Stub(False), Stub(True))
    assert has_runtime_lora(None, Stub(True))
