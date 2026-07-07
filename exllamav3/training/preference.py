"""
Preference-optimization losses (DPO / KTO) for QLoRA-on-EXL3.

Pure-torch loss functions over per-sequence completion log-probabilities, as
produced by ``NativeLlamaQLoRA.compute_logps`` (policy) and the same call under
``NativeLlamaQLoRA.adapters_disabled()`` (the frozen-base reference model --
the PEFT disable-adapter trick, so no second model is ever loaded).

Semantics follow HuggingFace TRL's stable trainers so hyperparameters and
results transfer directly:

  * DPO -- ``trl.DPOTrainer`` (Rafailov et al. 2023, arXiv:2305.18290), with
    the ``sigmoid`` (+ label smoothing, i.e. cDPO), ``hinge`` (SLiC-HF, Zhao
    et al. 2023) and ``ipo`` (Azar et al. 2023, arXiv:2310.12036) variants.
  * KTO -- ``trl.KTOTrainer`` (Ethayarajh et al. 2024, arXiv:2402.01306),
    promoted to TRL's stable API in huggingface/trl#6175, with the batch-level
    KL estimate from mismatched prompt/completion pairs and the
    ``apo_zero_unpaired`` variant (D'Oosterlinck et al. 2024).

Credit: the formulations here are adapted from HuggingFace TRL
(https://github.com/huggingface/trl, Copyright The HuggingFace Team,
Apache License 2.0). This is an independent reimplementation against the
exllamav3 native training path, not a copy of TRL code; see
``doc/qlora_handoff.md`` (Session 16) for the design notes.

Convention (matching TRL): a sequence's "logps" is the SUM of per-token
log-probabilities over its completion tokens only (prompt masked); rewards are
``beta * (policy_logps - reference_logps)``. All functions are shape-polymorphic
over a batch dimension and CPU-testable (``tests/test_preference.py``).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


DPO_LOSS_TYPES = ("sigmoid", "hinge", "ipo")
KTO_LOSS_TYPES = ("kto", "apo_zero_unpaired")


def dpo_loss(
    policy_chosen_logps: torch.Tensor,       # [b]
    policy_rejected_logps: torch.Tensor,     # [b]
    reference_chosen_logps: torch.Tensor,    # [b]
    reference_rejected_logps: torch.Tensor,  # [b]
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    chosen_counts: Optional[torch.Tensor] = None,    # [b] completion tokens (ipo)
    rejected_counts: Optional[torch.Tensor] = None,  # [b]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss over paired (chosen, rejected) rows.

    Returns ``(losses, chosen_rewards, rejected_rewards)``, each ``[b]``;
    rewards are detached (logging/accuracy only). ``loss_type``:

      * ``"sigmoid"`` (default): ``-log σ(beta·Δ)`` with ``Δ = (π_c - ref_c) -
        (π_r - ref_r)``; ``label_smoothing`` ε > 0 gives the cDPO/robust form
        ``-(1-ε)·log σ(beta·Δ) - ε·log σ(-beta·Δ)``.
      * ``"hinge"`` (SLiC): ``relu(1 - beta·Δ)``.
      * ``"ipo"``: ``(Δ̄ - 1/(2·beta))²`` on length-NORMALIZED logratios
        (``Δ̄`` uses per-token averages; pass ``chosen_counts`` /
        ``rejected_counts`` from ``compute_logps``).
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    if loss_type == "ipo":
        if chosen_counts is None or rejected_counts is None:
            raise ValueError("loss_type='ipo' needs chosen_counts/rejected_counts "
                             "(completion token counts) for length normalization")
        delta_avg = (chosen_logratios / chosen_counts.clamp(min=1)
                     - rejected_logratios / rejected_counts.clamp(min=1))
        losses = (delta_avg - 1.0 / (2.0 * beta)) ** 2
    else:
        delta = chosen_logratios - rejected_logratios
        if loss_type == "sigmoid":
            losses = (-(1.0 - label_smoothing) * F.logsigmoid(beta * delta)
                      - label_smoothing * F.logsigmoid(-beta * delta))
        elif loss_type == "hinge":
            losses = torch.relu(1.0 - beta * delta)
        else:
            raise ValueError(f"unknown DPO loss_type '{loss_type}' "
                             f"(expected one of {DPO_LOSS_TYPES})")

    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return losses, chosen_rewards, rejected_rewards


def kto_loss(
    policy_chosen_logps: torch.Tensor,        # [n_d]  (desirable rows)
    policy_rejected_logps: torch.Tensor,      # [n_u]  (undesirable rows)
    policy_kl_logps: Optional[torch.Tensor],  # [n_kl] mismatched pairs (or None)
    reference_chosen_logps: torch.Tensor,     # [n_d]
    reference_rejected_logps: torch.Tensor,   # [n_u]
    reference_kl_logps: Optional[torch.Tensor],  # [n_kl]
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    loss_type: str = "kto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kahneman-Tversky Optimization loss over UNPAIRED rows labeled
    desirable/undesirable (KTO paper eq. 7; TRL ``KTOTrainer`` semantics).

    Either subset may be empty for a given micro-batch (0-length tensors are
    fine). The reference point ``kl`` is the batch estimate of
    ``KL(π||ref)`` from MISMATCHED prompt/completion pairs (see
    :func:`mismatched_kl_shift`): ``mean(policy_kl_logps - reference_kl_logps)``,
    detached and clamped to ``>= 0``. Single-process estimate -- under DDP,
    all-reduce it across ranks before use (not wired here).

      * ``"kto"``:  chosen ``1 - σ(beta·(logratio - kl))``,
                    rejected ``1 - σ(beta·(kl - logratio))``.
      * ``"apo_zero_unpaired"``: no KL term; chosen ``1 - σ(beta·logratio)``,
                    rejected ``σ(beta·logratio)``.

    Returns ``(losses, chosen_rewards, rejected_rewards, kl)`` where ``losses``
    is the concatenation ``[desirable_weight·chosen ; undesirable_weight·
    rejected]`` (reduce with ``.mean()``/``.nanmean()``); rewards and ``kl``
    are detached.
    """
    if loss_type not in KTO_LOSS_TYPES:
        raise ValueError(f"unknown KTO loss_type '{loss_type}' "
                         f"(expected one of {KTO_LOSS_TYPES})")

    any_t = (policy_chosen_logps if policy_chosen_logps.numel()
             else policy_rejected_logps)
    if (loss_type == "kto" and policy_kl_logps is not None
            and policy_kl_logps.numel()):
        kl = (policy_kl_logps - reference_kl_logps).mean().detach().clamp(min=0)
        kl = kl.to(any_t.dtype)
    else:
        kl = any_t.new_zeros(())

    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    if loss_type == "kto":
        chosen_losses = 1.0 - torch.sigmoid(beta * (chosen_logratios - kl))
        rejected_losses = 1.0 - torch.sigmoid(beta * (kl - rejected_logratios))
    else:  # apo_zero_unpaired
        chosen_losses = 1.0 - torch.sigmoid(beta * chosen_logratios)
        rejected_losses = torch.sigmoid(beta * rejected_logratios)

    losses = torch.cat((desirable_weight * chosen_losses,
                        undesirable_weight * rejected_losses), dim=0)
    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return losses, chosen_rewards, rejected_rewards, kl


def mismatched_kl_shift(n: int) -> list[int]:
    """Index permutation pairing each prompt with ANOTHER example's completion
    for the KTO KL estimate: TRL's "+1 offset" rotation
    (``[n-1, 0, 1, ..., n-2]``, i.e. prompt i gets completion i-1). With
    ``n == 1`` the pair is matched, which would bias the KL estimate toward the
    policy's own reward -- callers should skip the KL term for singleton
    batches (the trainer warns; TRL requires batch size > 1 for the same
    reason)."""
    if n <= 0:
        return []
    return [n - 1] + list(range(n - 1))
