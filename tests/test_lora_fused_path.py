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
    # the BC bsz-1 graph fuses the whole MLP (gate/up/act/down), so it must
    # yield to an unfused branch when ANY of the three carries a LoRA
    assert re.search(
        r"self\.bc is not None and bsz == 1 and q_len == 1[^:]*?"
        r"not \(gu_lora or down_lora\)",
        src, re.S), "GatedMLP: BC bsz-1 branch lost its runtime-LoRA guard"


@pytest.mark.parametrize("fname", ["attn.py", "sliding_attn.py"])
def test_bc_attn_graph_dispatch_guarded(fname):
    # bc_attn/bc_swa graph-captured decode blocks run projections through
    # o_proj as one C++ call; the dispatch (not the cached graph build) must
    # check for a runtime LoRA on every involved projection
    src = _src("exllamav3", "modules", fname)
    assert re.search(
        r"bsz <= 4 and seqlen <= 16 and\s*"
        r"not has_runtime_lora\(self\.q_proj, self\.k_proj, self\.v_proj,",
        src), f"{fname}: graph-captured decode dispatch lost its runtime-LoRA guard"


def test_gdn_split_bsz1_graph_guarded():
    # BC_GatedDeltaNetSplit runs the whole layer in one call, reading qkv/z/o
    # trellis and the merged (base-weights-only) ba_weight_t buffer directly
    src = _src("exllamav3", "modules", "gated_delta_net.py")
    assert re.search(
        r"self\.bc_split and bsz == 1 and seqlen == 1[^:]*?"
        r"not has_runtime_lora\(self\.qkv_proj, self\.z_proj, self\.b_proj,",
        src, re.S), "GatedDeltaNet: split bsz-1 fused decode lost its runtime-LoRA guard"


def test_moe_expert_dispatch_guarded():
    # Every fused expert path must yield to the per-expert torch path (the
    # mlp() closure calling gates/ups/downs .forward) while an adapter is
    # loaded on any routed expert projection
    src = _src("exllamav3", "modules", "block_sparse_mlp.py")
    assert "experts_lora = has_runtime_lora(*self.gates, *self.ups, *self.downs)" in src
    # branch selection forces the torch-capable branch
    assert re.search(
        r"no_reconstruct or experts_lora or",
        src), "BlockSparseMLP: torch-path branch selection lost experts_lora"
    # exl3_moe fused kernel skipped under LoRA
    assert re.search(
        r"self\.fused_mode_buffers is not None and not experts_lora",
        src), "BlockSparseMLP: exl3_moe fused path lost its runtime-LoRA guard"
    # BC single-expert graph/DQ kernels skipped under LoRA
    assert re.search(
        r"self\.bc is not None and self\.support_quant_paths and not experts_lora",
        src), "BlockSparseMLP: BC single-expert path lost its runtime-LoRA guard"
    # bsz-1 graph (which may embed shared experts + shared gate) skipped when
    # the fused shared-expert linears carry a LoRA
    assert "sh_fused_lora" in src and re.search(
        r"elif self\.bc is not None and not sh_fused_lora",
        src), "BlockSparseMLP: bsz-1 graph lost its shared-experts LoRA guard"
    # raw-weight shared gate projection falls back to Linear.forward
    assert re.search(
        r"bsz > 32 or has_runtime_lora\(self\.shared_gate\)",
        src), "BlockSparseMLP: add_sigmoid_gate_proj lost its shared-gate LoRA guard"


def test_moe_expert_lora_slow_path_notice_present():
    # Expert adapters now apply via the unfused per-expert path; the loader
    # should still tell the user MoE decode will be slower while loaded.
    src = _src("exllamav3", "model", "lora.py")
    assert re.search(r"\\\.experts\\\.", src) and "per-expert" in src, \
        "lora.py: routed-expert slow-path notice removed"


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
