# <img src="doc/cat.png" width="40"> ExLlamaV3 — QLoRA training fork

**This is a fork of [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) that adds QLoRA fine-tuning directly on EXL3-quantized models.** Everything upstream does still works; on top of it there is a self-contained, **transformers-free** training path: a differentiable forward built over the EXL3 trellis (validated against the native inference forward to 100% argmax agreement), a plain-PyTorch trainer, and adapters that save in standard PEFT format. The original upstream README follows [below](#original-readme).

### Why train on EXL3 instead of bitsandbytes?

- Train on 2-8bpw quants, including non-integer. BNB is 4 or 8 bit.
- EXL3 is higher accuracy than BNB, theoretically this should help? - still needs benchmarking
- EXL3 is more performant in the lower and mid-size batches that are often used when squeezing big models onto consumer cards.

### Quick start

```bash
# install as usual (see upstream instructions below), plus the one training dep
pip install datasets            # optional extras: flash-attn, liger-kernel, bitsandbytes

# 1. prove the differentiable forward is correct for YOUR model (run this first)
python training/qlora_validate_native.py --model /path/to/exl3-model --compute-dtype bfloat16

# 2. edit the config, then train
python training/qlora_train.py --config training/qlora_train_config.yaml

# 3. before/after comparison on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3-model --adapter out/my_adapter
```

Everything is driven by one YAML file ([`training/qlora_train_config.yaml`](training/qlora_train_config.yaml) is the fully-commented reference — its keys mirror the CLI flags of `training/qlora_train_native.py` one-to-one). A minimal config looks like:

```yaml
model: /models/Llama-3.2-3B-Instruct-exl3-4bpw
out: out/my_adapter
parallel: split           # single | split (layer-split across GPUs) | ddp (torchrun)

r: 64
alpha: 64.0               # use alpha = r with init_lora: pissa
lr: 5e-6
epochs: 1.0
batch: 3
seq_len: 8192
pack: true                # sample packing (best-fit-decreasing, ~98% fill)

dataset: /data/my-set.jsonl
messages_key: messages    # OpenAI-style chats; or instruction/context/response keys
prompt_format: auto

init_lora: pissa          # default | pissa | qerr | eva  (see below)
eval_split: test          # held-out eval from the dataset's own split
eval_every: 50
compute_dtype: bfloat16
use_liger: true
optim: paged_adamw8bit
```

### What's in the box

- **Single-GPU, multi-GPU layer-split (`parallel: split`), and DDP (`parallel: ddp`)** training of LoRA adapters over a frozen EXL3 base — plus optional embedding/LM-head training (full or low-rank).
- **Preference optimization: DPO and KTO** (`training/qlora_train_pref.py --method dpo|kto`) with the frozen quantized base as the reference model (adapter-disable trick — no second model copy). Loss semantics follow [HuggingFace TRL](https://github.com/huggingface/trl)'s stable `DPOTrainer`/`KTOTrainer` (with credit — see below), so β/loss-variant hyperparameters transfer directly; variants: sigmoid/cDPO, hinge (SLiC), IPO, KTO, APO-zero-unpaired.
- **Memory levers** for long context on consumer cards: gradient checkpointing, activation offload to CPU RAM, fused/chunked cross-entropy (chunked over the vocab too, for 256k-vocab models), 8-bit and paged optimizers, Liger kernels (RMSNorm/RoPE/SwiGLU).
- **A real eval harness**: held-out loss from your dataset's own split (`eval_split`) or a carved fraction (`val_frac`), an optional second monitor set (`eval2_*`, e.g. wikitext LM loss watched next to your task loss), `save_best` checkpointing, periodic live sample generations, and a per-run CSV log of hyperparameters/losses/VRAM/throughput.
- **Correctness gates, not vibes**: `qlora_validate_native.py` checks the differentiable forward against the native inference forward, the Liger backward against plain torch, packing isolation, and each adapter init's step-0 math — before you spend GPU-days. A CPU test suite covers the gradient path end-to-end.
- **Standard outputs**: adapters save as PEFT-format safetensors, loadable by exllamav3's native LoRA loader (TabbyAPI), PEFT, or merge scripts.

### Modern PEFT: SVD adapter initializations

Short SFT runs spend a large fraction of their steps just growing the adapter off the ground (the default zero-init of B). This fork implements the current crop of SVD-based initializations, adapted to an immutable quantized base — select with one config key, `init_lora`:

- **`pissa`** ([PiSSA](https://arxiv.org/abs/2404.02948)) — the adapter starts as the top-r principal components of the base weights, trained against a residual base realized as a frozen offset (the trellis itself is never rewritten). Exports as a converted rank-2r standard LoRA so any consumer loads it correctly. **Current default recommendation: it won its first A/B clearly** (use `alpha = r`).
- **`eva`** ([EVA](https://arxiv.org/abs/2410.07170)) — A is initialized to the top-r right-singular vectors of each layer's *input activations*, streamed from your actual training data through the actual quantized forward in a short pre-pass. Function-preserving at step 0. Freshly built; being evaluated against pissa now.
- **`qerr`** (LoftQ-style, single-shot) — the adapter starts as the closest rank-r repair of the *quantization error* vs the original bf16 weights, aimed at the low-bpw regime where that error is large.
- **`use_rslora`** — rank-stabilized scaling (`alpha/sqrt(r)`) for rank sweeps.

Each init has a hard step-0 gate in `qlora_validate_native.py --init-lora <mode>`.

### Quantization-aware LoRA (`quant_aware`)

The deploy path for a trained adapter is merge-and-requantize, and ordinary QLoRA training is blind to it: the adapter is optimized against one fixed, exactly-known quantized base, so nothing stops it from relying on precision the requantize will destroy. Set `quant_aware` to make the *training forward* see that deploy-time uncertainty (adapted layers only; eval, checkpoints and validation always use exact weights):

- **`noise`** — fresh per-step pseudo-quantization noise on the frozen weights, scaled per output channel to each layer's quantization-error magnitude (the [NIPQ](https://arxiv.org/abs/2206.00820)-style differentiable proxy for the requantize the merged model will undergo).
- **`ste`** — the effective adapter delta is snapped to the quantization floor in the forward with a straight-through gradient: sub-floor delta components contribute nothing, exactly as after a requantize ([QA-LoRA](https://arxiv.org/abs/2309.14717)'s objective, rebuilt for a trellis base — its group-wise exact-merge operator needs affine zero-points that trellis quantization doesn't have).

The per-layer error scale is measured exactly against the original bf16 model when `quant_aware_ref_model` is set, else estimated from the trellis bitrate; `quant_aware_scale` multiplies it. Freshly built; being A/B-evaluated on the merge-and-requantize path now.

### Credits

The DPO/KTO preference-training implementation follows the loss semantics of **[HuggingFace TRL](https://github.com/huggingface/trl)**'s stable `DPOTrainer` and `KTOTrainer` (KTO stabilized in [trl#6175](https://github.com/huggingface/trl/pull/6175)). TRL is Apache-2.0 licensed, Copyright The HuggingFace Team; this fork reimplements the formulations independently against the EXL3 native training path rather than reusing TRL code. Underlying methods: DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)), KTO ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)), IPO, SLiC.

### Project status

Research fork under active development. The core mechanism is proven end-to-end: validated forward parity on the quantized weights, healthy trainings from 1B to 16B models on 1–2× RTX 3090 (including 8k-context packed runs on a 12B), adapters that load and steer generation on the native inference path. Training-side architecture support currently covers **Llama-family, Gemma 3/4, Qwen3-dense, Qwen3.5/3.6 dense hybrids (differentiable Gated DeltaNet + gated attention; box validation pending), and Mistral(-Nemo) dense models** (no MoE yet; no sample packing on Gated DeltaNet models); unsupported features are rejected loudly rather than silently mistrained. Interfaces may still move between sessions — the full engineering log with per-session results and rationale lives in [`doc/qlora_handoff.md`](doc/qlora_handoff.md), and experiment-specific tooling is quarantined in [`training/experiments/`](training/experiments/). Some inference-side fixes made here are candidates for upstreaming.

---

<a name="original-readme"></a>

# <img src="doc/cat.png" width="40"> ExLlamaV3

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs. Headline features:

- New [EXL3](doc/exl3.md) quantization format based on QTIP
- Flexible tensor-parallel and expert-parallel inference for consumer hardware setups
- OpenAI-compatible server provided via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) 
- Continuous, dynamic batching
- HF Transformers plugin (see [here](examples/transformers_integration.py))
- HF model support (see [supported architectures](#architecture-support))
- Speculative decoding
- 2-8 bit cache quantization
- Multimodal support
- LoRA support

The official and recommended backend server for ExLlamaV3 is [TabbyAPI](https://github.com/theroyallab/tabbyAPI/), which provides an OpenAI-compatible API for local or remote inference, with extended features like HF model downloading, embedding model support and support for HF Jinja2 chat templates.

### ⚠️ Important

- **Gemma4** does not currently support tensor/expert parallelism.

## Architecture support

- **AFM** (ArceeForCausalLM)
- **AfMoE** (AfmoeForCausalLM)
- **Apertus** (ApertursForCausalLM)
- **Command-R** etc. (CohereForCausalLM)
- **Command-A**, **Command-R7B**, **Command-R+** etc. (Cohere2ForCausalLM)
- **DeciLM**, **Nemotron** (DeciLMForCausalLM)
- **dots.llm1** (Dots1ForCausalLM)
- **ERNIE 4.5** (Ernie4_5_ForCausalLM, Ernie4_5_MoeForCausalLM)
- **EXAONE 4.0** (Exaone4ForCausalLM)
- **Gemma 2** (Gemma2ForCausalLM)
- **Gemma 3** (Gemma3ForCausalLM, Gemma3ForConditionalGeneration) *- multimodal*
- **Gemma 4** (Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration) *- multimodal* (E2B/E4B currently not supported)
- **GLM 4**, **GLM 4.5**, **GLM 4.5-Air**, **GLM 4.6** (Glm4ForCausalLM, Glm4MoeForCausalLM)
- **GLM 4.1V**, **GLM 4.5V** (Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration) *- multimodal*
- **HyperCLOVAX** (HyperCLOVAXForCausalLM, HCXVisionV2ForCausalLM) *- multimodal*
- **IQuest-Coder** (IQuestCoderForCausalLM)
- **LFM 2.5** (Lfm2MoeForCausalLM)
- **Llama**, **Llama 2**, **Llama 3**, **Llama 3.1-Nemotron** etc. (LlamaForCausalLM)
- **MiMo-RL** (MiMoForCausalLM)
- **MiniMax-M2** (MiniMaxM2ForCausalLM)
- **Mistral**, **Ministral 3**, **Devstral 2** etc. (MistralForCausalLM, Mistral3ForConditionalGeneration) *- multimodal*
- **Mixtral** (MixtralForCausalLM)
- **NanoChat** (NanoChatForCausalLM)
- **Olmo 3.1** (Olmo3ForCausalLM)
- **Olmo-Hybrid** (OlmoHybridForCausalLM)
- **Phi3**, **Phi4** (Phi3ForCausalLM)
- **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL** (Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration) *- multimodal*
- **Qwen 3** (Qwen3ForCausalLM, Qwen3MoeForCausalLM)
- **Qwen 3-Next** (Qwen3NextForCausalLM)
- **Qwen 3-VL** (Qwen3VLForConditionalGeneration)  *- multimodal*
- **Qwen 3-VL MoE** (Qwen3VLMoeForConditionalGeneration) *- multimodal*
- **Qwen 3.5** (Qwen3_5ForConditionalGeneration) *- multimodal*
- **Qwen 3.5 MoE** (Qwen3_5MoeForConditionalGeneration) *- multimodal*
- **Seed-OSS** (SeedOssForCausalLM)
- **SmolLM** (SmolLM3ForCausalLM)
- **SolarOpen** (SolarOpenForCausalLM)
- **Step 3.5 Flash** (Step3p5ForCausalLM)
- **Step 3.7 Flash** (Step3p7ForConditionalGeneration) *- multimodal*

Always adding more, stay tuned.


## What's missing?

Currently on the to-do list:

- ROCm support

As for what is implemented, expect that some things may be a little broken at first. Please be patient, raise issues and/or contribute. 👉👈 


## How to?

[TabbyAPI](https://github.com/theroyallab/tabbyAPI/) has a startup script that manages and installs prerequisites if you want to get started quickly with inference in an OAI-compatible client. 

Otherwise, start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Method 1: Installing from prebuilt wheel (recommended if you're unsure)

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Method 2: Installing from PyPi:

```sh
pip install exllamav3
```
Note that the PyPi package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).    

### Method 3: Building from source

Before building, make sure you have an appropriate version of Torch installed. Install a `flash-attn-2` wheel, e.g. from [here](https://mjunya.com/flash-attention-prebuild-wheels/). 

On Windows, you should also make sure you have the `triton-windows` package installed. ExLlamaV3 may work without it, but many things will work suboptimally.   

```sh
# Clone the repo
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3

# (Optional) switch to dev branch for latest in-progress features
git checkout dev

# Install requirements (make sure you install Torch separately)
pip install -r requirements.txt
```

At this point you should be able to run the conversion, eval and example scripts from the main repo directory, e.g. `python convert.py -i ...`

To install the library for the active venv, run from the repo directory:

```sh
pip install .
```

Relevant env variables for building:
- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for compilation. Set this to a reasonable value like 4 in that case.  
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch will build/load it at runtime instead.


## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors until the converted model can be compiled. It should have enough free space to store an entire copy of the output model. Note that while EXL2 conversion by default resumes an interrupted job when pointed to an existing folder, EXL3 needs you to explicitly resume with the `-r`/`--resume` argument.    

See [here](doc/convert.md) for more information.


## Examples

A number of example scripts are provided to showcase the features of the backend and generator. Some of them have hardcoded model paths and should be edited before you run them, but there is a simple CLI chatbot that you can start with:

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode> 

# E.g.:
python examples/chat.py -m /mnt/models/llama3.1-8b-instruct-exl3 -mode llama3

# Wealth of options
python examples/chat.py -h
```

## EXL3 quantization

<div align="center">
    <a href="doc/exl3.md" target="_blank">
        <img src="doc/llama31_8b_instruct_bpw.png" width="640">
    </a>
</div>

Despite their amazing achievements, most SOTA quantization techniques remain cumbersome or even prohibitively expensive to use. For instance, **AQLM** quantization of a 70B model takes around **720 GPU-hours** on an A100 server, costing $850 US at the time of writing. ExLlamaV3 aims to address this with the **EXL3** format, which is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. The conversion process is designed to be simple and efficient and requires only an input model (in HF format) and a target bitrate. By computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger ones (70B+) (on a single RTX 4090 or equivalent GPU.)

The [Marlin](https://github.com/IST-DASLab/marlin)-inspired GEMM kernel achieves roughly memory-bound latency under optimal conditions (4bpw, RTX 4090), though it still needs some work to achieve the same efficiency on Ampere GPUs and to remain memory-bound at lower bitrates.

Since converted models largely retain the original file structure (unlike **EXL2** which renames some tensors in its quest to turn every model into a Llama variant), it will be possible to extend **EXL3** support to other frameworks like HF Transformers and vLLM.

There are some benchmark results [here](doc/exl3.md), and a full writeup on the format is coming soon.

Fun fact: Llama-3.1-70B-EXL3 is coherent at 1.6 bpw. With the output layer quantized to 3 bpw and a 4096-token cache, inference is possible in under 16 GB of VRAM. 


### Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮  


### 🤗 HuggingFace repos

A selection of EXL3-quantized models is available [here](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a). Also shout out the following lovely people:
 
- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz) 
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode) 
- [Ready.Art](https://huggingface.co/ReadyArt) 
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
