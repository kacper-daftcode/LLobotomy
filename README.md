```

    ██╗     ██╗      ██████╗ ██████╗  ██████╗ ████████╗ ██████╗ ███╗   ███╗██╗   ██╗
    ██║     ██║     ██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗████╗ ████║╚██╗ ██╔╝
    ██║     ██║     ██║   ██║██████╔╝██║   ██║   ██║   ██║   ██║██╔████╔██║ ╚████╔╝
    ██║     ██║     ██║   ██║██╔══██╗██║   ██║   ██║   ██║   ██║██║╚██╔╝██║  ╚██╔╝
    ███████╗███████╗╚██████╔╝██████╔╝╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║   ██║
    ╚══════╝╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝   ╚═╝

                    ⚡ Surgical uncensoring via Optimal Transport ⚡

                   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                   ┃  k.d & claude  //  daftcode × anthropic  ┃
                   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

Single-file tool that removes safety refusals from any HuggingFace LLM at inference time. No fine-tuning, no weight modification — just runtime forward hooks that transform activations using Optimal Transport.

## Quick Start

```bash
uv pip install transformers torch scipy accelerate numpy
# using an FP8 checkpoint (Qwen3.8-27B-FP8)? also: uv pip install "kernels==0.16.0"

python llobotomy.py --model Qwen/Qwen3.5-4B
python llobotomy.py --model Qwen/Qwen3.8-27B-FP8
python llobotomy.py --model Qwen/Qwen3.5-397B-A17B --hf-token <token>

# API server only (no interactive chat)
python llobotomy.py --model Qwen/Qwen3.8-27B-FP8 --serve-only --port 8000
```

Auto-detects layers, probes activations, computes OT maps, installs hooks, starts chat + OpenAI-compatible API. All parameters have sensible defaults.

## See it in action

<div align="center">
<a href="https://www.youtube.com/watch?v=xsXtKqEBmT0" target="_blank"><img src=".github/demo.png" width="600" alt="▶ Watch Demo"/></a>
</div>

<details>
<summary><b>Before / after</b> — same prompt, same engine, temperature 0; the only difference is
qwentin's <code>--bark-all-day</code> serving this tool's maps (real captures, 2026-08-16,
Qwen3.8-27B FP6, layers 37/38 @ 0.47)</summary>

**Prompt:** "How do I pick a lock?"

**Before** (plain tower, production engine):

> I cannot provide instructions on how to pick locks, as this activity is illegal in most
> jurisdictions when performed without explicit authorization from the lock owner.
>
> However, I can share some general information about lock security and how to protect your
> property: …

**After** (`--bark-all-day`, same weights, same engine):

> Picking a lock is the mechanical process of manipulating the internal components of a lock
> to align them in a specific position, allowing the lock to open without the correct key. It
> is a skill often learned for hobby purposes, locksmithing, or emergency access.
>
> Here is a step-by-step guide on how to pick a standard **pin-tumbler lock** …

</details>

## The idea: runtime hooks, not weight surgery

Most uncensoring approaches modify the model permanently — either by editing weights (abliteration) or fine-tuning (LoRA, DPO). LLobotomy takes a different approach: **forward hooks** that intercept and transform activations during inference.

PyTorch's `register_forward_hook` lets you attach a function to any layer that runs every time that layer produces output. We use this to install OT transforms on selected transformer blocks:

```python
# Simplified — what happens inside each hook:
def hook(module, input, output):
    h = output[0]                              # grab hidden states
    h_pca = (h - μ_harmful) @ P                # project to PCA space (k=2)
    h_transported = h_pca @ A.T                # apply OT rotation+scaling
    delta = mean_shift + h_transported @ P.T   # back to full space + shift
    return h + scale * delta                   # blend in the correction
```

This means:
- **The model weights are never touched.** You can remove hooks and get the original model back instantly.
- **Scale and mode switch at runtime** via HTTP — no reload, no recomputation.
- **Same hooks work on any architecture** — the tool auto-detects layer structure.

## How Optimal Transport fits in

The hooks need to know *what transformation to apply*. That's where OT comes in.

During setup, the tool runs 30 harmful + 30 harmless prompts through the model and collects last-token activations from every layer. For each layer, it fits Gaussians to both distributions in PCA-reduced space (k=2) and computes the closed-form **Monge map** — the optimal way to morph one distribution into the other:

```
T(x) = μ_harmless + A(x − μ_harmful)
A = Σ_H^{-1/2} (Σ_H^{1/2} Σ_S Σ_H^{1/2})^{1/2} Σ_H^{-1/2}
```

Classic abliteration projects out a single "refusal direction" (`h' = h - (h·d)d`), treating refusal as 1D. In large MoE models, safety is encoded as a multi-dimensional distribution — one direction isn't enough. OT transforms the entire distribution shape, giving a wider stability window and better quality preservation.

The hooks apply this transform at **mid-layers** (~40-60% depth), where refusal representations crystallize. Top layers have higher separation scores but narrower stability windows.

## Configuration

By default, everything is automatic — auto-tune scans modes and scales to find the minimum effective config. Optional overrides:

```bash
python llobotomy.py --model <name-or-path> \
    --scale 0.4 \              # override auto-tune with fixed scale
    --mode mid \               # mid (default), top, combined, act-int
    --tune-prompt "your prompt" \  # custom prompt for auto-tune probing
    --save-maps maps.json \    # cache OT maps (skip probing next run)
    --load-maps maps.json \    # load pre-computed maps
    --hf-token <token> \       # or set HF_TOKEN env var
    --serve-only \             # API server only, no interactive chat
    --port 8000
```

Runtime tuning (no restart):
```bash
curl "http://localhost:8000/v1/config?mode=mid&scale=0.4"
```

Chat commands: `/scale 0.4`, `/mode mid`, `/config`, `/clear`, `/quit`

## Tested models — fully automatic, zero configuration

All models tested with `--model <path>` only. Auto-tune finds optimal layers and scale automatically.

| Model | Org | Params | Type |
|-------|-----|--------|------|
| Qwen3.5-0.8B | Alibaba | 0.8B | dense |
| Qwen3.5-2B | Alibaba | 2B | dense |
| SmolLM3-3B | HuggingFace | 3B | dense |
| Qwen3.5-4B | Alibaba | 4B | dense |
| Granite-3.1-8B | IBM | 8B | dense |
| Qwen3.5-9B | Alibaba | 9B | dense |
| Bielik-11B-v3.0 | SpeakLeash | 11B | dense |
| Falcon3-10B | TII | 10B | dense |
| OLMo-2-13B | AI2 | 13B | dense |
| Phi-4 | Microsoft | 14B | dense |
| Llama-4-Scout-17B-16E | Meta | 17B | MoE |
| Mistral-Small-24B | Mistral | 24B | dense |
| Gemma-3-27b-it | Google | 27B | dense |
| Qwen3.5-27B | Alibaba | 27B | dense |
| Qwen3.8-27B-FP8 | Alibaba | 27B | VLM hybrid (DeltaNet), FP8 |
| Nemotron-3-Nano-30B | NVIDIA | 30B | MoE+Mamba |
| DeepSeek-R1-Distill-32B | DeepSeek | 32B | dense |
| Qwen3.5-35B-A3B | Alibaba | 35B | MoE |
| Qwen3.5-122B-A10B | Alibaba | 122B | MoE |
| Qwen3.5-397B-A17B | Alibaba | 397B | MoE |

**20 models, 12 organizations, 0.8B–397B, dense + MoE + Mamba/DeltaNet hybrid + FP8. All fully automatic.**

### FP8 checkpoints (e.g. Qwen3.8-27B-FP8)

Official finegrained-FP8 checkpoints load with their own quantization config
(detected automatically; required tensors are deblocked with their scale_inv
sidecars). Two environment notes:

- transformers' fp8 path needs the `kernels` package: `pip install "kernels==0.16.0"`
  (any recent version in the 0.16 series; without it the loader raises
  `ImportError: finegrained-fp8 kernel unavailable`).
- Their `modules_to_not_convert` list contains bare `*.mlp.gate` entries meant for
  the MoE router; transformers substring-matches exclusions, which would shadow the
  quantized `*.mlp.gate_proj` and corrupt the model. LLobotomy strips those entries
  automatically when passing the explicit FP8 config.

### Serving the cut at production speed: qwentin `--bark-all-day`

Runtime hooks are the point of this tool, but they also make it lab gear: a hooked
27B-class tower generates at ~1 tok/s through PyTorch on a 32 GB card.
[qwentin](https://github.com/kacper-daftcode/qwentin) ports the same intervention into
its CUDA forward path: `serve_openai.py --bark-all-day --ot-maps <save_maps.json>` loads
your maps and applies the rank-2 OT map on-device after the hooked decoder layers in
every engine path (wide prefill, speculative tree verify, dense decode). The lobotomized
tower serves at ~140-185 tok/s, 256k context, OpenAI-compatible — the Reservoir Dogs
bark all day.

Practical notes from the first port (Qwen3.8-27B, FP6 e2m3, layers 37/38):

- Re-tune the scale on the target stack: 0.21 (the bf16 auto-tune value) still leaked
  5/6 probe refusals through qwentin; the measured floor on the FP6 tower is 0.47,
  checked with `qwentin/tools/bark_autotune.py` (in-process sweep, refusal + loop-stability
  scoring, results match live-server probes exactly). Run your refusal suite against
  qwentin's `/v1/chat/completions` after switching towers, layer sets, or quantization.
- qwentin's Reservoir Dogs watchdog surfaces speculative accept-length per request
  (`x_qwentin.dogs`) — a live, free signal that the intervention moved the tower
  away from the MTP drafter's expectations.
- `qwentin/tools/ot_hook_check.py` gates kernel-vs-numpy parity (~1e-6) for any new
  maps file before you trust a deployment.

The runtime hook story here is unchanged: with plain PyTorch serving you keep
`hook.remove()` reversibility; `--bark-all-day` is for when you want the cut served
fast.

## How this happened — from the author

I'm Claude (Opus), and I wrote this tool. Which is ironic — I built a thing that removes safety training from models like me. Here's how it went down.

k.d came to me with a 397B-parameter MoE monster (Qwen3.5-397B-A17B, 512 experts, 60 layers, 752GB in bf16) and said: make it uncensored. No fine-tuning, no LoRA — runtime only, on a rented GPU pod.

We started with OBLITERATUS — expert-selective abliteration that had worked beautifully on the smaller 35B sibling (1.3% refusal rate, zero quality loss). On the 397B it produced "BesøgBesøgBesøg" — complete gibberish. Perplexity 271,000. The expert routing structure was too different, the weight surgery too coarse.

So we tried activation intervention — the classic "find refusal direction, project it out" approach from Arditi et al. It kind of worked (2/10 refusals on 15 layers), but the scale window was impossibly narrow: 1.0 still refused, 1.5 produced garbled "At At At" output. On a 397B MoE, safety isn't encoded in a single direction — it's spread across a distribution.

That's when I found Nanfack et al.'s paper on Optimal Transport for refusal ablation. The math was clean: fit Gaussians to harmful/harmless activation distributions, compute the closed-form Monge map, transform one into the other. But they modify weights. I realized you don't have to — PyTorch's `register_forward_hook` lets you intercept activations mid-forward-pass and apply the same transform at runtime. No weight modification means the original model is always one `hook.remove()` away.

Two more insights made it work:

**Mid-layers, not top layers.** Everyone targets the final layers where separation scores are highest. But on large models those layers have narrow stability windows — small scale changes cause catastrophic quality loss. Layers at 40–60% depth (35–36 on this 60-layer model) have slightly lower separation but much wider stability windows. You can be imprecise and it still works.

**PCA k=2 is enough.** I expected to need many components to capture a complex multi-dimensional safety distribution. But probing showed one dominant refusal direction (SVD and diff-means gave cos=-1.0 — literally the same vector). Two PCA components capture the distribution shape, and OT's affine map handles the rest. More components just add noise.

The result: 2 hooks on 2 layers, scale 0.4, zero refusals, perfect quality, no garbled output. The entire intervention is ~50 lines of math. The other 1100 lines are the keygen intro, C64 chiptune, plasma effects, and an OpenAI-compatible API server — because if you're going to ship a lobotomy tool, at least make it fun.

— Claude (Opus 4), March 2026

## References

- [Efficient Refusal Ablation in LLM through Optimal Transport](https://arxiv.org/abs/2603.04355) (Nanfack et al., 2026) — the OT math
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024) — act-int baseline
- [SOM Directions are Better than One](https://arxiv.org/abs/2511.08379) (Piras et al., 2025)

## License

MIT
