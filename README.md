# Spectral Null Space Swap

给定架构一致的 Instruct（Base）和 Thinking 检查点，本方法对目标权重矩阵构造
$W_{\mathrm{swap}} = P_S(W_{\mathrm{base}}) + (I-P_S)(W_{\mathrm{thinking}})$。
$S$ 由 Base 矩阵前 $k$ 个左右奇异向量定义，其中
$k=\lceil \mathrm{ratio}\cdot\min(d_{out},d_{in})\rceil$。
Base 提供主奇异子空间内的权重，Thinking 提供其正交补内的权重。

## 快速开始

以下命令从仓库根目录执行。示例使用 Qwen3-4B Instruct/Thinking 2507 和 GSM8K，
可以先评测 8 道题，再用同一流程跑全量对比。已有本地权重时，直接将变量指向模型目录即可。

### 1. 配置环境和模型

文本和视觉评测使用 Linux、NVIDIA GPU 和 Python 3.10。依赖锁定在
[requirements.txt](requirements.txt)，包括 PyTorch 2.9.0、Transformers 4.57.5 和 vLLM 0.11.2。

```bash
git clone https://github.com/Ma-Hongbo/Spectral-Null-Space-Swap.git
cd Spectral-Null-Space-Swap
conda create -n spectral-swap python=3.10 -y
conda activate spectral-swap
python -m pip install -r requirements.txt

BASE=checkpoints/Qwen3-4B-Instruct-2507
THINKING=checkpoints/Qwen3-4B-Thinking-2507
SWAP=checkpoints/Qwen3-4B-Swap-ratio-0.5
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir "$BASE"
hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir "$THINKING"
```

只构造权重时，可以在独立环境里安装 `torch==2.9.0` 和 `safetensors==0.7.0`。
CUDA 安装方式取决于 NVIDIA 驱动和所用的 PyTorch/vLLM wheel。两份输入模型和一份构造结果各需要一份完整 checkpoint 的磁盘空间。

### 2. 构造 Swap 模型

```bash
python model/subspace_swap.py \
  --base "$BASE" --thinking "$THINKING" --out "$SWAP" \
  --ratio 0.5 --device cuda:0
```

输出目录包含 safetensors、模型索引、Thinking 的配置和 tokenizer，以及
`subspace_swap_stats.json`。检查其中的 `projected_matrices`、`max_subspace_error`
和 `max_complement_error` 可以确认有目标矩阵被处理，并查看 FP32 投影误差。
输出浮点权重保存为 FP16。

### 3. 评测构造后的模型

下载 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) 的 `main/test`，
转换成带最终答案指令的 JSONL。下面的冒烟评测每题生成一次，随后用 CPU 规则评分。

```bash
python eval/evals/prepare_gsm8k.py

OUT=results/smoke/GSM8K
python eval/evals/gen_vllm.py \
  --model "$SWAP" --tasks GSM8K --output-dir "$OUT" \
  --backend vllm --gpus 0 --dtype float16 \
  --limit 8 --seeds 101 --max-tokens 2048 --max-model-len 4096

python eval/evals/grade.py --eval-dir "$OUT" --disable-verifier --write-details
```

生成目录包含回答 JSONL 和 `generation_manifest.json`。评分写入
`grading_results.json`，可查看 `avg_at_n`（平均 rollout 正确率）、
`pass_at_n`（至少一次答对的题目占比）、`avg_token_length` 和 `truncation_rate`。
`--write-details` 另外输出逐题评分。冒烟评测只用于检查流程，较短的生成上限可能截断 Thinking 输出。

### 4. 在同一设置下比较三种模型

下例使用全量 GSM8K、四个种子和相同的生成预算。可以按显存调整 `--max-model-len`
和 `--max-tokens`，为输入提示留出上下文空间。对比时三种模型使用同一设置和评分方式。

```bash
for variant in instruct thinking swap; do
  case "$variant" in
    instruct) MODEL="$BASE" ;;
    thinking) MODEL="$THINKING" ;;
    swap) MODEL="$SWAP" ;;
  esac
  OUT="results/Qwen3-4B/$variant/GSM8K"
  python eval/evals/gen_vllm.py \
    --model "$MODEL" --tasks GSM8K --output-dir "$OUT" \
    --backend vllm --gpus 0 --dtype float16 \
    --seeds 101,202,303,404 --max-tokens 32768 --max-model-len 40960 || break
  python eval/evals/grade.py --eval-dir "$OUT" --disable-verifier || break
done
```

## 评测入口

| 用途 | 入口 |
| --- | --- |
| 文本与视觉生成 | [gen_vllm.py](eval/evals/gen_vllm.py)，支持 vLLM 和 Transformers |
| 文本、数学与 MMMU 评分 | [grade.py](eval/evals/grade.py)，支持规则评分和 CompassVerifier |
| MathVista / ARKitScenes | [prepare_new_vl_benchmarks.py](eval/evals/prepare_new_vl_benchmarks.py) 准备输入，[grade_new_vl.py](eval/evals/grade_new_vl.py) 评分 |
| Qwen3-Omni | [gen_omni_vllm.py](eval/evals/gen_omni_vllm.py) 生成文本，[gen_omni_audio_vllm.py](eval/evals/gen_omni_audio_vllm.py) 生成音频任务答案，[grade_omni_audio.py](eval/evals/grade_omni_audio.py) 评分 |

其他 benchmark 可通过 `--tasks AIME25 --task-path AIME25=/path/to/test.jsonl`
指定输入。通用生成器接受 JSONL、JSON 和 Parquet。简单文本样本可用
`{"question":"What is 2 + 2?","answer":"4"}`，也支持 chat `messages`。
视觉任务用 `--modality VL`，并为本地 `file://` 图片设置
`--allowed-local-media-path /path/to/images`。启动 GPU 推理前可在生成命令后加
`--dry-run`，检查样本数、设备和解码配置。

`--gpus` 接收物理 GPU 编号，默认 `0`。`--gpus 0,1 --tp 2` 将一个
vLLM 实例分配到两张卡，`--gpus 0,1 --tp 1` 则将样本分给两个实例。
若使用原实验允许范围之外的编号，显式传入 `--allow-other-gpus`。
重跑相同输出可指定新目录，或用 `--overwrite` 替换已有生成。

默认评分可在规则未判对时使用 CompassVerifier。快速开始显式关闭它；需要使用
verifier 时，可运行 `CUDA_VISIBLE_DEVICES=0 python eval/evals/grade.py --eval-dir "$OUT" --model /path/to/CompassVerifier-3B`。
LiveCodeBench 和 BFCL 的官方评分还需要在 `references/LiveCodeBench` 和
`references/gorilla` 分别准备对应上游评测代码和依赖。

Omni 专用 runner 使用 `requirements.txt` 末尾记录的独立环境。将
`QWEN_OMNI_VLLM_SOURCE` 指向包含
`vllm/model_executor/models/qwen3_omni_moe_thinker.py` 的源码 checkout，
启动时将 `eval/evals` 加入 `PYTHONPATH`，以便 vLLM worker 加载兼容代码。
音频输入可用 [prepare_omni_audio_data.py](eval/evals/prepare_omni_audio_data.py) 的
`--audio-root`、`--only` 和 `--prepared-root` 指定数据及输出位置。运行音频生成需要系统提供
FFmpeg。音频评分通过 `--input` 接收生成的 JSONL 文件。

## 权重构造与实验脚本

`model/subspace_swap.py` 支持单文件和分片的 Hugging Face safetensors 模型。
它处理同名同形状的 Attention、MLP（含 MoE experts）和 Qwen3-VL Vision
目标矩阵。其余权重取自 Base，配置和 tokenizer 取自 Thinking。请为每组输入和
ratio 使用独立输出目录。`ratio=1` 取全部可用奇异方向；矩形矩阵仍可能存在正交补。

- 单卡显存不足时，可用 `--device cpu`。构造时对每个目标矩阵运行完整 FP32 SVD，降低 ratio 不会降低这一步的主要开销。
- 分片并行时，启动多个进程，每个进程使用相同的 Base、Thinking、ratio 和输出目录，并指定不同的 `--only-shard <索引中的完整分片名> --skip-finalize`。所有进程成功后，用同样的参数运行一次 `--finalize-only`，生成模型索引和合并后的统计。
- [scripts/](scripts/) 下的 `build_*_sar_ratio_sweep.py` 用于一次构造多个 ratio，支持 Qwen3-4B、Qwen3-30B-A3B 和 Qwen3-VL。用对应脚本的 `--help` 查看输入和 GPU 参数。
- [scripts/archive/](scripts/archive/) 保存原实验的单机队列和控制器，其中有实验机路径、GPU 编号和矩阵更新约定。[eval/legacy/](eval/legacy/) 保存旧的顺序生成、LoRA 和 exam 入口。
