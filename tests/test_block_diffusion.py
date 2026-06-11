import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pytest
import torch
import torch.testing

from exllamav3.generator.block_diffusion import (
    BlockDiffusionSettings,
    eb_accept_mask,
    token_entropy,
)

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)


@pytest.mark.parametrize("shape", [(1, 16), (1, 256), (4, 256)])
@pytest.mark.parametrize("scale", [0.1, 1.0, 30.0])
def test_token_entropy_matches_categorical(shape, scale):
    torch.manual_seed(0)
    logits = torch.randn(*shape, 512) * scale
    ref = torch.distributions.Categorical(logits = logits).entropy()
    ent = token_entropy(logits)
    torch.testing.assert_close(ent, ref, rtol = 1e-4, atol = 1e-5)


def _eb_accept_reference(entropy: torch.Tensor, bound: float) -> torch.Tensor:
    # Brute force per row: accept the largest k lowest-entropy positions such that
    # sum(first k) - max(first k) <= bound
    mask = torch.zeros_like(entropy, dtype = torch.bool)
    for row in range(entropy.shape[0]):
        ent = entropy[row]
        order = torch.argsort(ent, descending = False)
        best_k = 0
        for k in range(1, len(order) + 1):
            head = ent[order[:k]]
            if head.sum() - head.max() <= bound:
                best_k = k
            else:
                break
        mask[row, order[:best_k]] = True
    return mask


@pytest.mark.parametrize("bound", [0.05, 0.1, 1.0, 100.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eb_accept_matches_reference(bound, seed):
    torch.manual_seed(seed)
    entropy = torch.rand(3, 64) * 2.0
    mask = eb_accept_mask(entropy, bound)
    ref = _eb_accept_reference(entropy, bound)
    assert torch.equal(mask, ref)


def test_eb_accept_always_accepts_one():
    entropy = torch.full((1, 32), 100.0)
    mask = eb_accept_mask(entropy, 0.001)
    assert mask.sum().item() == 1


def test_eb_accept_low_entropy_all():
    entropy = torch.zeros((1, 32))
    mask = eb_accept_mask(entropy, 0.1)
    assert mask.all()


def test_temperature_schedule_endpoints():
    s = BlockDiffusionSettings()
    n = s.max_denoising_steps
    first = s.t_min + (s.t_max - s.t_min) * (n / n)
    last = s.t_min + (s.t_max - s.t_min) * (1 / n)
    assert first == pytest.approx(s.t_max)
    assert last == pytest.approx(s.t_min + (s.t_max - s.t_min) / n)


def test_settings_from_directory(tmp_path):
    gc = {
        "max_denoising_steps": 32,
        "t_min": 0.3,
        "t_max": 0.9,
        "stability_threshold": 2,
        "confidence_threshold": 0.01,
        "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.2},
        "eos_token_id": [1, 106],
    }
    with open(tmp_path / "generation_config.json", "w") as f:
        json.dump(gc, f)
    s = BlockDiffusionSettings.from_directory(str(tmp_path))
    assert s.max_denoising_steps == 32
    assert s.t_min == 0.3
    assert s.t_max == 0.9
    assert s.stability_threshold == 2
    assert s.confidence_threshold == 0.01
    assert s.entropy_bound == 0.2


def test_settings_defaults_without_file(tmp_path):
    s = BlockDiffusionSettings.from_directory(str(tmp_path))
    assert s == BlockDiffusionSettings()
