# Mechanistic Interpretability & Analysis (`interpretability/`)

Code for the mechanistic and ablation analyses in **$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient** — covering Section 4.1 (*Energy Partition*), Section 5 (*Mechanistic Analysis*), and Appendix A (*Mechanistic Analysis Details*).

Only analysis and plotting code is tracked in Git; generated intermediate measurements (`*/data/*.json`) and rendered figures (`*.pdf`, `*.png`) are written locally at runtime.

| Directory | Paper Section | Focus |
| --- | --- | --- |
| [`entropy/`](entropy/) | §5.1, Appendix A.1 / A.2 / A.4 | Attention-entropy concentration, top-$k$ attention weights, per-problem scatter, and pretraining entropy saturation |
| [`corollary_check/`](corollary_check/) | §5.2, Appendix A | Empirical verification of the first-order condition $\langle \Delta W_\perp, G_\perp \rangle < 0$ across 7 reasoning benchmarks |
| [`hallucination/`](hallucination/) | Appendix A.3 | Factual accuracy, abstention, and hallucination rates on SimpleQA & TruthfulQA (plus MoE probe builder) |
| [`norm_decomposition/`](norm_decomposition/) | §4.1 (Fig. 1) | Parameter-space vs. function-space energy decomposition and random-subspace controls |

---

## Probe Family & Notation

Let $W_0$ denote the **Instruct (Base)** checkpoint and $W_t$ the paired **Thinking** checkpoint from the same model family. With post-training update $\Delta W = W_t - W_0$ and orthogonal projector $P_S$ onto the full leading singular subspace of $W_0$ ($\rho = 1$), the update decomposes orthogonally as:

$$
\Delta W = \Delta W_\parallel + \Delta W_\perp, \qquad \Delta W_\parallel = P_S(\Delta W), \quad \Delta W_\perp = (I - P_S)(\Delta W).
$$

This defines the four probe checkpoints compared throughout the mechanistic analysis:

| Variant | Code Key | Weight Definition | Interpretation |
| --- | --- | --- | --- |
| **Base** | `base` | $W_0$ | Original Instruct checkpoint |
| **Sub** | `sub` / `subonly` | $W_0 + \Delta W_\parallel$ | In-subspace component of the Thinking update only |
| **Null** | `perp` / `perponly` | $W_0 + \Delta W_\perp$ | Null-space component only (equivalent to $S^3$ at $\rho = 1$) |
| **Full** | `full` | $W_0 + \Delta W = W_t$ | Original Thinking checkpoint |

---

## Setup & Environment Configuration

Install the main repository environment ([`../requirements.txt`](../requirements.txt)), which includes `torch`, `transformers`, `safetensors`, `vllm`, `datasets`, `numpy`, and `matplotlib`.

Machine-specific checkpoint and workspace locations are configured via two environment variables so scripts contain no hardcoded cluster paths:

| Variable | Meaning | Default |
| --- | --- | --- |
| `S3_ROOT` | Working directory containing `paths2507.env`, `data/`, and derived probe `models/` | `.` |
| `S3_HF_HOME` | Optional Hugging Face cache override (`HF_HOME`) | `~/.cache/huggingface` |

```bash
export S3_ROOT=/path/to/workdir
export S3_HF_HOME=/path/to/hf-cache   # optional
```

Create `$S3_ROOT/paths2507.env` pointing to your local checkpoints:

```sh
# $S3_ROOT/paths2507.env
BASE2=/path/to/Qwen3-4B-Instruct-2507
RL2=/path/to/Qwen3-4B-Thinking-2507
BASE0=/path/to/Qwen3-4B-Base          # optional; only used by dist_from_base.py
```

Individual scripts also accept CLI overrides such as `--data`, `--src`, and `--out` where applicable.

---

## 1. Attention Entropy & Pretraining Saturation (`entropy/`)

Measures head-wise attention entropy concentration across layers 32–35 for **Base**, **Sub**, **Null**, and **Full**, as well as attention-entropy saturation across Pythia pretraining checkpoints.

| Script | Role |
| --- | --- |
| [`gen_cot.py`](entropy/gen_cot.py) | Generates fixed Chain-of-Thought traces with `Qwen3-4B-Thinking-2507` on AIME problems |
| [`ent_cot.py`](entropy/ent_cot.py) | Sweeps Pythia-1.4B pretraining checkpoints (`step0`–`step143000`) to measure layer-wise entropy and row-space gradient ratios |
| [`plot_fig5_combined.py`](entropy/plot_fig5_combined.py) | Plots the 3-panel main figure: multi-dataset entropy bars, L32–35 top-8 attention heatmaps, and per-problem scatter |
| [`plot_figA1_layerwise.py`](entropy/plot_figA1_layerwise.py) | Plots layer-wise entropy profiles and ordering consistency across layers |
| [`plot_figA2_8datasets.py`](entropy/plot_figA2_8datasets.py) | Plots per-dataset attention-entropy bar charts across 8 benchmarks |
| [`plot_saturation.py`](entropy/plot_saturation.py) | Plots the pretraining entropy-saturation trajectory |
| [`plot_scatter_all_datasets.py`](entropy/plot_scatter_all_datasets.py) | Plots per-problem entropy scatter ($H_{\mathrm{Base}}$ vs. $H_{\mathrm{model}}$) across datasets |

> **Note on JSON keys:** Plotting scripts read JSON summaries from `entropy/data/` keyed as `{variant}_{dataset}_{layer}` with `variant` in `{base, sub, perp, full}` (`perp` denotes the **Null** variant).

---

## 2. First-Order Corollary Verification (`corollary_check/`)

Tests Proposition 5.2 / Corollary 5.4 empirically: the entropy ordering $H_{\mathrm{Null}} < H_{\mathrm{Base}} < H_{\mathrm{Sub}}$ holds at first order when the null-space update $\Delta W_\perp$ has a negative inner product with $G_\perp$, where $G = \nabla_W H\big|_{W_0}$ is the gradient of mean L32–35 attention entropy evaluated at $W_0$.

| Script | Role |
| --- | --- |
| [`grad_cos.py`](corollary_check/grad_cos.py) | Single-dataset reference check across all 36 layers $\times$ 7 projection matrices |
| [`grad_cos_7datasets.py`](corollary_check/grad_cos_7datasets.py) | **Main runner** — computes $\langle \Delta W_\perp, G_\perp \rangle$ and concatenated cosines across 7 math benchmarks (`aime24`, `aime25`, `amc23`, `cmimc25`, `hmmt25`, `math500`, `gsm8k`) |
| [`grad_cos_perprob.py`](corollary_check/grad_cos_perprob.py) | Memory-efficient per-problem inner-product breakdown |
| [`own_cot_check.py`](corollary_check/own_cot_check.py) | Evaluates gradient alignment on each variant's own CoT traces alongside direct entropy differences |
| [`run_7datasets.sh`](corollary_check/run_7datasets.sh) | Launches one `grad_cos_7datasets.py` worker per dataset across GPUs with sufficient free VRAM |
| [`plot_cos_ip_bar.py`](corollary_check/plot_cos_ip_bar.py) | Renders the paper's bar/arcsecond deviation figure from `--src` (defaults to `corollary_check/data/grad_cos_7datasets`) |
| [`plot_cos_ip.py`](corollary_check/plot_cos_ip.py) | Alternative line / broken-axis visualization |

```bash
# Single dataset on one GPU
python interpretability/corollary_check/grad_cos_7datasets.py --datasets aime24 --pool 12 --nseq 10

# Multi-GPU sweep across all 7 datasets + summary report
bash interpretability/corollary_check/run_7datasets.sh
python interpretability/corollary_check/grad_cos_7datasets.py --report

# Render figure from computed JSON directory
python interpretability/corollary_check/plot_cos_ip_bar.py --src "$S3_ROOT/grad_cos_7datasets"
```

### Key Implementation Details

- **Memory-saving projector identity:** Because $P_S$ is self-adjoint and idempotent and $\Delta W_\perp \in S^\perp$, $\langle \Delta W_\perp, G_\perp \rangle = \langle \Delta W_\perp, G \rangle$ holds identically. Precomputing $\Delta W_\perp$ once avoids storing or projecting per-matrix gradient buffers on GPU (reducing peak VRAM from ~31 GB to ~17 GB in `grad_cos_perprob.py`).
- **Concatenated cosine vs. sign:** Individual flattened weight matrices have $2.6\text{M}$–$25\text{M}$ entries, so meaningful alignment is measured on the full concatenated vector against the $1/\sqrt{D}$ random baseline (plotting $90^\circ - \text{angle}$ in arcseconds), with the negative sign consistently carried by MLP blocks.

---

## 3. Hallucination Evaluation (`hallucination/`)

Evaluates factual precision, abstention (`NOT_ATTEMPTED`), and hallucination rate ($\text{incorrect} / \text{attempted}$) across the four probe variants on SimpleQA and TruthfulQA.

| Script | Role |
| --- | --- |
| [`prepare_data.py`](hallucination/prepare_data.py) | Downloads TruthfulQA (`multiple_choice`) and OpenAI SimpleQA into `hallucination/data/` |
| [`build_perponly_moe.py`](hallucination/build_perponly_moe.py) | Constructs `perp` ($W_0 + \Delta W_\perp$) and `subonly` ($W_0 + \Delta W_\parallel$) checkpoints for dense and MoE architectures |
| [`eval_simpleqa.py`](hallucination/eval_simpleqa.py) | Runs vLLM generation on SimpleQA and grades normalized `\boxed{}` answers |
| [`run_hallucination.py`](hallucination/run_hallucination.py) | Multi-GPU job dispatcher across `{base, subonly, perponly, full}` |
| [`analyze_results.py`](hallucination/analyze_results.py) | Aggregates accuracy, abstention, and hallucination rates with per-category breakdowns |

---

## 4. Parameter- vs. Function-Space Norm Decomposition (`norm_decomposition/`)

Compares how much of $\Delta W$ resides in $S$ vs. $S^\perp$ in **parameter space** ($f_*^{(\ell)} = \|\Delta W_*^{(\ell)}\|_F^2 / \|\Delta W^{(\ell)}\|_F^2$) versus **function space** ($\tilde{f}_*^{(\ell)} = \|\Delta h_*^{(\ell)}\|^2 / \|\Delta h_{\mathrm{full}}^{(\ell)}\|^2$, measured via hidden-state displacements $\Delta h_*^{(\ell)}(x) = h_{W_0 + \Delta W_*}^{(\ell)}(x) - h_{W_0}^{(\ell)}(x)$), alongside random-subspace controls.

| Script | Role |
| --- | --- |
| [`dist_from_base.py`](norm_decomposition/dist_from_base.py) | Computes displacement norms and cosines of Instruct and Thinking checkpoints relative to Pretrained Base |
| [`dist_check.py`](norm_decomposition/dist_check.py) | Computes pairwise Frobenius distances among `Base`, `Sub`, `Null`, and `Full` |
| [`compute_functional_norms.py`](norm_decomposition/compute_functional_norms.py) | Streams hidden states to compute layer-wise function-space shares $\tilde{f}_{\mathrm{sub}}$, $\tilde{f}_{\mathrm{perp}}$, $\tilde{f}_{\mathrm{rand}}$, and nonlinear interaction residuals on AIME prompts |
| [`compute_functional_multi.py`](norm_decomposition/compute_functional_multi.py) | Extends function-space norm decomposition across MATH-500, GSM8K, AMC23, and SimpleQA |
| [`build_randpar.py`](norm_decomposition/build_randpar.py) | Builds the dimension-matched random-subspace control checkpoint (`randpar_2507`) |
| [`rand_subspace_control.py`](norm_decomposition/rand_subspace_control.py) | Tests spectral alignment significance against the random-basis expectation $r^2 / (mn)$ |
| [`plot_norm_combined.py`](norm_decomposition/plot_norm_combined.py) | Renders the 3-panel energy-partition and layer-wise norm figure from `norm_decomposition/data/{weight_norms,functional_norms}.json` |
