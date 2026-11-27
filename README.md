# Spectral Nullspace Swap

本仓库用于在两个结构一致的 Hugging Face 模型检查点之间执行 spectral
subspace swap。给定 Base 模型权重 $W_{base}$ 和 Thinking 模型权重
$W_{thinking}$，脚本构造：

$$
W_{mix}=P_S(W_{base})+(I-P_S)(W_{thinking}),
$$

其中 $S$ 由 $W_{base}$ 的前 $k$ 个左右奇异向量定义，
$k=\lceil ratio\cdot\min(d_{out},d_{in})\rceil$。

该方法在 Base 的主奇异子空间内保留 Base 权重，在其正交补空间内使用
Thinking 权重。Thinking 权重及其更新不会进行额外的低秩截断。

## 目录结构

```text
spectral_nullspace_swap/
├── model/
│   └── subspace_swap.py
├── eval/
│   └── evals/
├── scripts/
├── requirements.txt
└── README.md
```

## 环境配置

推荐使用 Conda 创建独立环境。主 Text/VL 实验记录的环境为 Python
3.10.19、PyTorch 2.9.0、Transformers 4.57.5 和 vLLM 0.11.2。

```bash
conda create -n spectral-swap python=3.10.19 -y
conda activate spectral-swap
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果只运行 `subspace_swap.py`，不进行生成和评测，只需要安装核心依赖：

```bash
python -m pip install torch==2.9.0 safetensors==0.7.0
```

GPU 版本的 PyTorch 需要与机器上的 NVIDIA 驱动及 CUDA 环境兼容。如果集群
使用特定的 PyTorch wheel，请先按照集群配置安装 `torch==2.9.0`，再安装
`requirements.txt` 中的其余依赖。

Qwen3-Omni 音频评测使用另一套独立环境，不能与上述 vLLM 环境混装；具体版本
已经记录在 `requirements.txt` 末尾。仅构造模型权重时不需要 Omni 环境。

## 输入模型要求

`--base` 和 `--thinking` 必须指向本地 Hugging Face 模型目录，并满足：

- 两个模型结构一致，待处理权重的名称和形状一致；
- 权重采用 `safetensors` 格式；
- 支持单文件 `model.safetensors`；
- 支持带有 `model.safetensors.index.json` 的分片检查点；
- 输出目录需要至少容纳一个完整模型检查点。

脚本处理以下线性权重：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`；
- MLP：`gate_proj`、`up_proj`、`down_proj`，包括 MoE experts；
- Qwen3-VL Vision：`qkv`、`proj`、`linear_fc1`、`linear_fc2`。

Embedding、Norm、LM head、router 和其他非目标权重直接保留 Base 模型权重。
模型配置、tokenizer 和 chat template 等元数据从 Thinking 模型目录复制。

## 基本用法

在仓库根目录运行：

```bash
python model/subspace_swap.py \
  --base /path/to/base-model \
  --thinking /path/to/thinking-model \
  --out /path/to/output-model-ratio-0.5 \
  --ratio 0.5 \
  --device cuda:0
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--base` | Base/Instruct 模型目录 |
| `--thinking` | Thinking 模型目录 |
| `--out` | 输出模型目录；不同 ratio 应使用不同目录 |
| `--ratio` | Base 主奇异子空间的秩比例，范围为 `(0, 1]`，默认 `0.10` |
| `--device` | 执行 FP32 SVD 和投影的设备，默认 `cuda:0`；也可设为 `cpu` |
| `--only-shard` | 只构造指定权重分片，可重复传入 |
| `--skip-finalize` | 只写权重分片和 worker 统计，不生成最终索引 |
| `--finalize-only` | 所有分片完成后，仅生成索引、配置和合并统计 |

`ratio` 越大，使用 Base 权重的主子空间越大；`ratio=1` 时，所有目标矩阵
都退化为 Base 权重。较小的 ratio 会在更多方向上保留 Thinking 权重。

计算过程使用 FP32，最终权重以 FP16 保存。

## 分片并行构造

对于分片模型，可以让不同 GPU 分别处理不同的 safetensors shard。首先查看
Base 模型的分片名称：

```bash
ls /path/to/base-model/*.safetensors
```

然后启动多个 worker。所有 worker 必须使用相同的 Base、Thinking、输出目录和
ratio，每个 worker 负责不同的 shard：

```bash
CUDA_VISIBLE_DEVICES=0 python model/subspace_swap.py \
  --base /path/to/base-model \
  --thinking /path/to/thinking-model \
  --out /path/to/output-model-ratio-0.5 \
  --ratio 0.5 \
  --device cuda:0 \
  --only-shard model-00001-of-00002.safetensors \
  --skip-finalize &

CUDA_VISIBLE_DEVICES=1 python model/subspace_swap.py \
  --base /path/to/base-model \
  --thinking /path/to/thinking-model \
  --out /path/to/output-model-ratio-0.5 \
  --ratio 0.5 \
  --device cuda:0 \
  --only-shard model-00002-of-00002.safetensors \
  --skip-finalize &

wait
```

全部 worker 完成后执行一次 finalize：

```bash
python model/subspace_swap.py \
  --base /path/to/base-model \
  --thinking /path/to/thinking-model \
  --out /path/to/output-model-ratio-0.5 \
  --ratio 0.5 \
  --finalize-only
```

只有在所有预期分片均已生成后，finalize 才会成功。

## 输出文件

成功完成后，输出目录包含：

- 构造后的 `*.safetensors` 权重分片；
- `model.safetensors.index.json`；
- 从 Thinking 模型复制的配置和 tokenizer 文件；
- `subspace_swap_stats.json`，记录整体参数和数值误差；
- 分片并行时产生的 `subspace_swap_stats_*.json` worker 统计。

统计文件中的 `subspace_error` 和 `complement_error` 用于检查目标分解是否满足：

$$
P_S(W_{mix})=P_S(W_{base}),
$$

$$
(I-P_S)(W_{mix})=(I-P_S)(W_{thinking}).
$$

## 常见问题

### CUDA out of memory

脚本会对每个目标矩阵执行完整 FP32 SVD。降低 `ratio` 不会减少完整 SVD 的主要
显存开销。可以减少同一 GPU 上的并发 worker 数量，换用显存更大的 GPU，或者
使用 `--device cpu`，但 CPU SVD 会明显更慢。

### Shape mismatch

这通常表示 Base 和 Thinking 不是同一模型架构或不是可逐层对应的检查点。请
确认两个目录中的模型规模、层数、hidden size 和 MoE 配置一致。

### Unknown safetensors shard

`--only-shard` 必须使用 Base 模型索引中记录的完整文件名。可从
`model.safetensors.index.json` 或模型目录中的文件名确认。
