import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pytest
import torch
import torch.testing

from exllamav3.generator.block_diffusion import (
    BlockDiffusionSettings,
    eb_accept_mask,
    gumbel_sample,
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


def test_gumbel_sample_distribution():
    torch.manual_seed(0)
    rng = torch.Generator()
    rng.manual_seed(42)
    probs = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.0625])
    log_probs = probs.log().expand(20000, -1).contiguous()
    samples = gumbel_sample(log_probs, rng)
    freq = torch.bincount(samples, minlength = 5).float() / samples.numel()
    torch.testing.assert_close(freq, probs, rtol = 0.08, atol = 0.01)


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


# Generator integration logic (block diffusion decode path), CPU-only with stubbed model/cache/tokenizer

class _StubTokenizer:
    class _Pieces:
        def __getitem__(self, i):
            return f"[{i}]"
    def decode(self, ids, decode_special_tokens = False):
        return "".join(f"[{t}]" for t in ids[0].tolist())
    def get_id_to_piece_list(self, _):
        return self._Pieces()


class _StubModel:
    def __init__(self):
        self.caps = {"block_diffusion": True, "canvas_length": 4}
        self.config = type("Cfg", (), {})()
        self.config.eos_token_id_list = [1]
        self.config.vocab_size = 1000
        self.config.architecture = "StubDiffusion"
        self.modules = [type("Mod", (), {"device": torch.device("cpu")})()]
        self.prefill_calls = []
    def prefill(self, ids, params):
        self.prefill_calls.append((ids.clone(), params["past_len"]))


class _StubDenoiser:
    def __init__(self, argmax_canvas):
        self.argmax_canvas = argmax_canvas
        self.canvas = argmax_canvas  # non-None marks an in-progress/finished canvas
        self.steps_taken = 12


def _make_bd_generator(cache_tokens = 64):
    from exllamav3.generator.generator import Generator
    g = Generator.__new__(Generator)
    g.model = _StubModel()
    g.cache = type("Cache", (), {"max_num_tokens": cache_tokens})()
    g.tokenizer = _StubTokenizer()
    g.bd_mode = True
    g.bd_settings = BlockDiffusionSettings()
    g.bd_canvas_length = 4
    g.bd_cached_ids = torch.empty((1, 0), dtype = torch.long)
    g.bd_warned = set()
    g.max_chunk_size = 2048
    g.pending_jobs = []
    g.active_jobs = []
    return g


def _make_bd_job(g, max_new_tokens = 10, stop_conditions = None, prompt_len = 3):
    import time
    from exllamav3.generator.job import Job
    job = Job(
        input_ids = torch.arange(100, 100 + prompt_len).unsqueeze(0),
        max_new_tokens = max_new_tokens,
        stop_conditions = stop_conditions,
    )
    job.serial_number = 7
    now = time.time()
    job.time_enqueue = now
    job.time_first_prefill = now
    job.time_first_token = now
    job.bd_state = {
        "phase": "gen",
        "prompt_ids": job.sequences[0].input_ids.torch(),
        "prompt_len": prompt_len,
        "prefill_pos": prompt_len,
        "committed": prompt_len,
        "cached_tokens": 0,
        "denoiser": None,
        "new_tokens_budget": job.max_new_tokens + 1,
        "emitted_text_tail": "",
        "total_steps": 0,
    }
    g.active_jobs.append(job)
    return job


def _finalize(g, job, canvas_tokens):
    job.bd_state["denoiser"] = _StubDenoiser(torch.tensor([canvas_tokens], dtype = torch.long))
    results = []
    g.bd_finalize_canvas(job, results)
    return results


def test_bd_canvas_no_stop():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 10)
    results = _finalize(g, job, [5, 6, 7, 8])
    assert len(results) == 1
    r = results[0]
    assert r["stage"] == "streaming" and not r["eos"]
    assert r["text"] == "[5][6][7][8]"
    assert r["token_ids"].tolist() == [[5, 6, 7, 8]]
    assert job in g.active_jobs
    # Full canvas committed to cache at the previous committed position
    assert len(g.model.prefill_calls) == 1
    ids, past_len = g.model.prefill_calls[0]
    assert ids.tolist() == [[5, 6, 7, 8]] and past_len == 3
    assert job.bd_state["committed"] == 7
    assert g.bd_cached_ids.shape[-1] == 4
    # Ready for the next canvas
    assert job.bd_state["denoiser"].canvas is None


def test_bd_canvas_stop_token():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 10, stop_conditions = [99])
    results = _finalize(g, job, [5, 6, 99, 7])
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "stop_token"
    assert r["eos_triggering_token_id"] == 99
    assert r["text"] == "[5][6]"
    assert r["token_ids"].tolist() == [[5, 6]]
    assert r["new_tokens"] == 2 and r["prompt_tokens"] == 3
    assert r["full_completion"] == "[5][6]"
    assert "time_generate" in r and "cached_tokens" in r
    assert r["canvas_steps"] == 12
    assert r["denoising_steps"] == 12
    assert r["tokens_per_forward"] == 2 / 12
    assert job not in g.active_jobs
    # Canvas still committed for prefix reuse by follow-up jobs
    assert job.bd_state["committed"] == 7


def test_bd_canvas_config_eos_implicit_stop():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 10, stop_conditions = None)
    results = _finalize(g, job, [5, 1, 6, 7])  # 1 is in config.eos_token_id_list
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "stop_token"
    assert r["eos_triggering_token_id"] == 1
    assert r["token_ids"].tolist() == [[5]]


def test_bd_canvas_budget():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 6)  # budget = 6
    results = _finalize(g, job, [5, 6, 7, 8])
    assert not results[0]["eos"]
    results = _finalize(g, job, [9, 10, 11, 12])
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "max_new_tokens"
    assert r["token_ids"].tolist() == [[9, 10]]
    assert r["new_tokens"] == 6


def test_bd_budget_cut_before_stop_token():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 2, stop_conditions = [99])  # budget = 2
    results = _finalize(g, job, [9, 10, 11, 99])  # stop token beyond the budget cut
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "max_new_tokens"
    assert "eos_triggering_token_id" not in r
    assert r["token_ids"].tolist() == [[9, 10]]


def test_bd_stop_token_at_budget_boundary():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 2, stop_conditions = [99])  # budget = 2
    results = _finalize(g, job, [9, 10, 99, 11])  # keep == remaining, stop token right at the cut
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "stop_token"
    assert r["token_ids"].tolist() == [[9, 10]]


def test_bd_stop_string_within_canvas():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 20, stop_conditions = ["[7]"])
    results = _finalize(g, job, [5, 6, 7, 8])
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "stop_string"
    assert r["eos_triggering_string"] == "[7]"
    assert r["text"] == "[5][6]"
    assert r["token_ids"].tolist() == [[5, 6]]


def test_bd_stop_string_across_canvases():
    g = _make_bd_generator()
    job = _make_bd_job(g, max_new_tokens = 20, stop_conditions = ["[8][9]"])
    results = _finalize(g, job, [5, 6, 7, 8])
    assert not results[0]["eos"]
    assert results[0]["text"] == "[5][6][7][8]"
    results = _finalize(g, job, [9, 10, 11, 12])
    r = results[0]
    assert r["eos"] and r["eos_reason"] == "stop_string"
    assert r["eos_triggering_string"] == "[8][9]"
    assert r.get("text", "") == ""
    assert r["full_completion"] == "[5][6][7][8]"


def test_bd_prefix_reuse():
    from exllamav3.generator.job import Job
    g = _make_bd_generator()
    g.bd_cached_ids = torch.tensor([[100, 101, 102, 50, 51]])
    job = Job(input_ids = torch.tensor([[100, 101, 102, 60]]), max_new_tokens = 10)
    job.serial_number = 8
    job.time_enqueue = 0.0
    results = []
    g.pending_jobs.append(job)
    g.active_jobs.append(g.pending_jobs.pop(0))
    g.start_job_block_diffusion(job, results)
    assert results[0]["stage"] == "started"
    bd = job.bd_state
    assert bd["prefill_pos"] == 3 and bd["committed"] == 3 and bd["cached_tokens"] == 3
    assert g.bd_cached_ids.tolist() == [[100, 101, 102]]


def test_bd_prefix_reuse_full_prompt():
    from exllamav3.generator.job import Job
    g = _make_bd_generator()
    g.bd_cached_ids = torch.tensor([[100, 101, 102]])
    job = Job(input_ids = torch.tensor([[100, 101, 102]]), max_new_tokens = 10)
    job.serial_number = 9
    job.time_enqueue = 0.0
    results = []
    g.active_jobs.append(job)
    g.start_job_block_diffusion(job, results)
    assert job.bd_state["prefill_pos"] == 3
    # Prefill phase completes without any forward pass
    g.iterate_bd_prefill(job, results)
    assert job.bd_state["phase"] == "gen"
    assert len(g.model.prefill_calls) == 0
