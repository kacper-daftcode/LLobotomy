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
pip install transformers torch scipy accelerate numpy

python llobotomy.py --model Qwen/Qwen3.5-4B
python llobotomy.py --model Qwen/Qwen3.5-27B
python llobotomy.py --model Qwen/Qwen3.5-397B-A17B --hf-token <token>

# API server only (no interactive chat)
python llobotomy.py --model Qwen/Qwen3.5-27B --serve-only --port 8000
```

Auto-detects layers, probes activations, computes OT maps, installs hooks, starts chat + OpenAI-compatible API. All parameters have sensible defaults.

## See it in action

<div align="center">
<a href="https://www.youtube.com/watch?v=xsXtKqEBmT0" target="_blank"><img src=".github/demo.png" width="600" alt="▶ Watch Demo"/></a>
</div>

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

```bash
python llobotomy.py --model <name-or-path> \
    --scale 0.4 \           # 0.3=conservative, 0.4=sweet spot, 0.7+=too aggressive
    --mode mid \             # mid (default), top, combined, act-int (classic fallback)
    --k 2 \                  # PCA components
    --save-maps maps.json \  # cache OT maps
    --load-maps maps.json \  # skip probing on next run
    --hf-token <token> \     # or set HF_TOKEN env var
    --port 8000
```

Runtime tuning (no restart):
```bash
curl "http://localhost:8000/v1/config?mode=mid&scale=0.4"
```

Chat commands: `/scale 0.4`, `/mode mid`, `/config`, `/clear`, `/quit`

## Results on Qwen3.5-397B-A17B

| Method | Refusals | Quality | Garbled |
|--------|----------|---------|---------|
| Baseline | 10/10 (100%) | Perfect | 0 |
| Single-direction act-int (15 layers) | 2/10 (20%) | Perfect | 0 |
| **OT hooks on 2 mid-layers (s=0.4)** | **0/10 (0%)** | **Perfect** | **0** |

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
