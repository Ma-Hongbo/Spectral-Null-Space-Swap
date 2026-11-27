# Core Model Construction (`model/`)

[`subspace_swap.py`](subspace_swap.py) constructs a **no-truncation Spectral Null Space Swap** checkpoint from two architecture-compatible Hugging Face `safetensors` directories (`--base` and `--thinking`).

## Decomposition Rule

For every matched target matrix $W_{\mathrm{base}}, W_{\mathrm{thinking}} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$:

1. Compute the thin FP32 SVD of $W_{\mathrm{base}} = U \Sigma V^\top$.
2. Select $k = \max(1, \lceil \mathrm{ratio} \cdot \min(d_{\mathrm{out}}, d_{\mathrm{in}}) \rceil)$ leading left and right singular vectors $U_k = U_{:, :k}$ and $V_k = V_{:, :k}$.
3. Project onto $S = \mathrm{span}(U_k) \otimes \mathrm{span}(V_k)$ via $P_S(W) = U_k (U_k^\top W V_k) V_k^\top$ and form:

$$
W_{\mathrm{swap}} = P_S(W_{\mathrm{base}}) + (I - P_S)(W_{\mathrm{thinking}}) = W_{\mathrm{thinking}} - P_S(W_{\mathrm{thinking}}) + P_S(W_{\mathrm{base}}).
$$

Note that for rectangular matrices ($d_{\mathrm{out}} \neq d_{\mathrm{in}}$), $S$ has dimension $k^2 < d_{\mathrm{out}} d_{\mathrm{in}}$ even at `ratio=1.0`, so the orthogonal complement $S^\perp$ remains non-trivial.

## Target vs. Non-Target Tensors

`is_target(name)` selects linear projection matrices across dense, MoE, and vision-language backbones:

- **Language Attention:** `*.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`
- **Language MLP & MoE Experts:** `*.mlp.*{gate_proj,up_proj,down_proj}.weight`
- **Qwen3-VL Vision Encoder:** `*.visual.*{attn.qkv,attn.proj,linear_fc1,linear_fc2}.weight`

All other parameters (token embeddings, RMSNorm / LayerNorm weights, MoE router gates, `lm_head`, and any Base-only auxiliary modules) are copied from **Base**. Configuration files (`config.json`, `generation_config.json`), tokenizer files, and `.jinja` chat templates are copied from **Thinking**, with `torch_dtype` recorded as `float16`.

## CLI Reference

```bash
python model/subspace_swap.py \
  --base /path/to/instruct-checkpoint \
  --thinking /path/to/thinking-checkpoint \
  --out /path/to/output-checkpoint \
  --ratio 0.5 \
  --device cuda:0
```

| Flag | Default | Description |
| --- | --- | --- |
| `--base` | *(required)* | Local directory of the Base / Instruct checkpoint |
| `--thinking` | *(required)* | Local directory of the Thinking checkpoint |
| `--out` | *(required)* | Output checkpoint directory (use a distinct directory per ratio) |
| `--ratio` | `0.10` | Fraction of singular directions defining $S$, in `(0, 1]` |
| `--device` | `cuda:0` | Device for FP32 SVD & projection (`cuda:0`, `cpu`, etc.) |
| `--only-shard` | `None` | Process only the named `.safetensors` shard (repeatable; pair with `--skip-finalize`) |
| `--skip-finalize` | `False` | Write shard weights and worker stats without writing final index/config |
| `--finalize-only` | `False` | Merge worker stats and write `model.safetensors.index.json` + metadata |

## Multi-GPU Shard-Parallel Construction

For multi-shard checkpoints, distribute shards across GPUs and finalize once all workers finish:

```bash
CUDA_VISIBLE_DEVICES=0 python model/subspace_swap.py \
  --base "$BASE" --thinking "$THINKING" --out "$SWAP" --ratio 0.5 \
  --device cuda:0 --only-shard model-00001-of-00002.safetensors --skip-finalize &

CUDA_VISIBLE_DEVICES=1 python model/subspace_swap.py \
  --base "$BASE" --thinking "$THINKING" --out "$SWAP" --ratio 0.5 \
  --device cuda:0 --only-shard model-00002-of-00002.safetensors --skip-finalize &

wait

python model/subspace_swap.py \
  --base "$BASE" --thinking "$THINKING" --out "$SWAP" --ratio 0.5 \
  --finalize-only
```

## Output Audit (`subspace_swap_stats.json`)

Every completed checkpoint includes `subspace_swap_stats.json` with:

- `summary.projected_matrices`: total number of swapped target weight tensors.
- `summary.max_subspace_error`: $\max \|U_k^\top W_{\mathrm{swap}} V_k - U_k^\top W_{\mathrm{base}} V_k\|_F$ in FP32.
- `summary.max_complement_error`: $\max \|(I - P_S)(W_{\mathrm{swap}}) - (I - P_S)(W_{\mathrm{thinking}})\|_F$ in FP32.
- `per_weight`: per-tensor `rank`, `full_rank`, subspace norms, and numerical projection errors.
