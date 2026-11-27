# Multi-Ratio Sweep Builders & Merging Baselines (`scripts/`)

While [`model/subspace_swap.py`](../model/subspace_swap.py) builds a single checkpoint at one `--ratio`, the scripts in this directory support **single-pass multi-ratio sweeps** (reusing each matrix's SVD across all ratios) and **streaming TIES-Merging baselines**.

All scripts can be invoked directly from the repository root.

---

## 1. Multi-Ratio Spectral Null Space Swap Builders

Computing the full FP32 SVD of each weight matrix dominates build time. These builders compute `torch.linalg.svd` **once per target matrix** and emit checkpoints for every requested ratio (default `0.9,0.8,0.7,0.6,0.5`) in a single shard-by-shard pass. They import target-layer selection (`is_target`) and shard indexing (`build_tensor_map`) directly from `model/subspace_swap.py`.

### Qwen3-4B Sweep

Writes `<output-root>/qwen3-4b-sar-ratio-<ratio>/` in `float16`:

```bash
python scripts/build_qwen3_4b_sar_ratio_sweep.py \
  --base checkpoints/Qwen3-4B-Instruct-2507 \
  --thinking checkpoints/Qwen3-4B-Thinking-2507 \
  --output-root checkpoints/sweeps/qwen3-4b \
  --ratios 0.9,0.8,0.7,0.6,0.5 \
  --device cuda:0
```

### Qwen3-30B-A3B & Qwen3-Omni-30B-A3B Sweep

Shard-resumable (`--resume`), preserves source floating-point `dtype`, and safely handles Qwen3-Omni where the Instruct checkpoint includes Talker / Code2Wav weights absent from the thinker-only Thinking checkpoint:

```bash
python scripts/build_qwen3_30b_a3b_sar_ratio_sweep.py \
  --base checkpoints/Qwen3-30B-A3B-Instruct-2507 \
  --thinking checkpoints/Qwen3-30B-A3B-Thinking-2507 \
  --output-root checkpoints/sweeps/qwen3-30b-a3b \
  --ratios 0.9,0.8,0.7,0.6,0.5 \
  --device cpu \
  --threads 32 \
  --resume
```

### Qwen3-VL Sweep

Swaps both language-model and vision-encoder (`qkv`, `proj`, `linear_fc1`, `linear_fc2`) target matrices, writing `<output-root>/<output-prefix>-<ratio>/`:

```bash
python scripts/build_qwen3_vl_sar_ratio_sweep.py \
  --base checkpoints/Qwen3-VL-4B-Instruct \
  --thinking checkpoints/Qwen3-VL-4B-Thinking \
  --output-root checkpoints/sweeps/qwen3-vl-4b \
  --output-prefix Qwen3-VL-4B-sar-ratio \
  --ratios 0.9,0.8,0.7,0.6,0.5 \
  --device cpu \
  --threads 32 \
  --resume
```

---

## 2. Streaming TIES-Merging Baseline

[`ties_merge_safetensors.py`](ties_merge_safetensors.py) implements [TIES-Merging](https://arxiv.org/abs/2306.01708) (`trim -> mass sign election -> disjoint mean`) over Hugging Face `safetensors` checkpoints in two streaming passes via `safe_open`, avoiding materializing 30B models in host RAM.

```bash
python scripts/ties_merge_safetensors.py \
  --base checkpoints/Qwen3-4B-Base \
  --models checkpoints/Qwen3-4B-Instruct-2507 checkpoints/Qwen3-4B-Thinking-2507 \
  --metadata-source checkpoints/Qwen3-4B-Thinking-2507 \
  --output checkpoints/Qwen3-4B-TIES-d0.2 \
  --density 0.20 \
  --scale 1.0 \
  --resume
```

Outputs a validated `safetensors` checkpoint along with `ties_merge_stats.json` recording trim thresholds, sign-election statistics, and shard validation results.
