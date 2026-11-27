# Evaluation Pipeline (`eval/`)

The evaluation suite is organized by pipeline stage so data preparation, GPU inference, and grading can run independently. By default, prepared benchmark datasets live under `eval/data/` (ignored by Git).

```text
eval/
├── prepare/    # Download / normalize raw datasets into JSONL + dataset_manifest.json
├── generate/   # vLLM & Transformers inference for Text, Math, Code, and Vision-Language
├── omni/       # Isolated Qwen3-Omni text & audio runners + vLLM 0.9.2 compatibility plugin
└── grade/      # Symbolic math, multiple-choice, 3D-IoU, external, and verifier-backed graders
```

---

## 1. Dataset Preparation (`eval/prepare/`)

### GSM8K (`prepare_gsm8k.py`)

Downloads `openai/gsm8k` (`main`, `test` split) and writes `eval/data/GSM8K/test.jsonl` with a `\boxed{}` final-answer instruction:

```bash
python eval/prepare/prepare_gsm8k.py
```

### Vision-Language: MathVista & ARKitScenes (`prepare_new_vl_benchmarks.py`)

Converts `AI4Math/MathVista` (`testmini` parquet) and Omni3D's `ARKitScenes` annotations into OpenAI-style multimodal JSONL with local `file://` image URIs:

```bash
# MathVista-testmini only
python eval/prepare/prepare_new_vl_benchmarks.py \
  --only-mathvista \
  --mathvista-parquet /path/to/testmini.parquet \
  --mathvista-image-dir eval/data/VL_MathVista/images \
  --mathvista-output eval/data/VL_MathVista/testmini.jsonl

# ARKitScenes (9-DoF 3D bounding-box detection) only
python eval/prepare/prepare_new_vl_benchmarks.py \
  --only-arkitscenes \
  --omni3d-annotation /path/to/ARKitScenes_test.json \
  --arkit-image-root /path/to/omni3d/images \
  --arkit-output eval/data/VL_ARKitScenes/test.jsonl
```

### Omni Audio: MMAU, MMAR, and MMSU (`prepare_omni_audio_data.py`)

Extracts audio files and normalizes metadata under `eval/data/omni_audio/prepared/{MMAU,MMAR,MMSU}/samples.jsonl`:

```bash
python eval/prepare/prepare_omni_audio_data.py \
  --audio-root /path/to/raw_audio_benchmarks \
  --prepared-root eval/data/omni_audio/prepared \
  --only all
```

---

## 2. Text & Vision-Language Generation (`eval/generate/gen_vllm.py`)

`gen_vllm.py` supports JSONL, JSON, and Parquet benchmarks, running four reproducible rollouts per example (`--seeds 101,202,303,404`) by default and writing `<task>_*.jsonl` plus `generation_manifest.json`.

### Built-in Task Registry & Custom Tasks

- **Registered tasks** (resolved relative to `--data-dir`, default `eval/data`): `GSM8K`, `AIME24`, `AIME25`, `AIME26`, `AMC23`, `MATH-500`, `Minerva`, `Olympiad-Bench`, `BRUMO25`, `CMIMC25`, `HMMT25`, `MMLU-Pro`, `ARC-C`, `MathVista-testmini`, `ARKitScenes`.
- **Custom paths** (`LiveCodeBench`, `BFCL`, `MMLU`, `MMMU`, or any custom JSONL/Parquet): supply `--task-path TASK=/path/to/file.jsonl`. Minimal custom text rows only need `{"question": "What is 2 + 2?", "answer": "4"}` (or chat `messages`).

### Common Examples

```bash
# Dry-run check (no GPU initialization)
python eval/generate/gen_vllm.py \
  --model "$SWAP" --tasks AIME25 --task-path AIME25=/path/to/aime25.parquet \
  --output-dir results/AIME25 --dry-run

# Multi-GPU data-parallel text generation (2 engines x TP=1 on GPUs 0 and 1)
python eval/generate/gen_vllm.py \
  --model "$SWAP" --tasks GSM8K --output-dir results/GSM8K \
  --gpus 0,1 --tp 1 --dtype float16

# Vision-Language generation on MathVista-testmini
python eval/generate/gen_vllm.py \
  --model checkpoints/Qwen3-VL-4B-sar-ratio-0.5 \
  --tasks MathVista-testmini \
  --modality VL \
  --allowed-local-media-path eval/data/VL_MathVista/images \
  --output-dir results/VL/Qwen3-VL-4B/swap/MathVista-testmini \
  --gpus 0 --tp 1
```

> **GPU Allow-List Note:** If you pass GPU `4` or device IDs `>= 10` to `--gpus`, add `--allow-other-gpus`.

---

## 3. Qwen3-Omni Generation (`eval/omni/`)

Activate the isolated `requirements/omni.txt` environment (`Python 3.11`, `vLLM 0.9.2`) and point `QWEN_OMNI_VLLM_SOURCE` to a checkout containing `vllm/model_executor/models/qwen3_omni_moe_thinker.py`:

```bash
export QWEN_OMNI_VLLM_SOURCE=/path/to/vllm-qwen3-omni
export PYTHONPATH="$PWD/eval/omni:${PYTHONPATH:-}"
```

```bash
# Resumable Omni text generation (MMLU or GSM8K, requires >= 2 GPUs)
python eval/omni/gen_omni_vllm.py \
  --model checkpoints/qwen3-30b-a3b-sar-ratio-0.5 \
  --benchmark GSM8K \
  --data eval/data/GSM8K/test.jsonl \
  --output-dir results/OMNI/swap/GSM8K \
  --gpus 0,1

# Resumable Omni audio generation (requires ffmpeg on PATH)
python eval/omni/gen_omni_audio_vllm.py \
  --model checkpoints/qwen3-30b-a3b-sar-ratio-0.5 \
  --benchmark MMAU \
  --variant swap \
  --data eval/data/omni_audio/prepared/MMAU/samples.jsonl \
  --output-dir results/OMNI_AUDIO/swap/MMAU \
  --gpus 0,1 \
  --official-settings
```

[`gen_omni.py`](omni/gen_omni.py) is also provided as a pure `transformers` (`device_map="auto"`) runner for text-only MMLU when vLLM is not used.

---

## 4. Grading (`eval/grade/`)

### Text, Math, MMMU, LiveCodeBench & BFCL (`grade.py`)

```bash
# Fast CPU symbolic/rule grading (SymPy + mathd normalization)
python eval/grade/grade.py --eval-dir results/GSM8K --disable-verifier --write-details

# Rule grading + CompassVerifier-3B fallback for unmatched responses
python eval/grade/grade.py \
  --eval-dir results/GSM8K \
  --model opencompass/CompassVerifier-3B \
  --gpus 0

# Recursive grading across an entire results tree
python eval/grade/grade.py --results-root results/Qwen3-4B --disable-verifier
```

- **External official checkers:** For `LiveCodeBench` and `BFCL`, clone the upstream repositories under `references/LiveCodeBench` and `references/gorilla` at the repository root so `grade.py` can invoke their official test/state evaluators automatically.

### MathVista & ARKitScenes (`grade_new_vl.py`)

Automatically dispatches between official MathVista answer-type normalization and Omni3D ARKitScenes 9-DoF oriented 3D-box `mAP@0.15`:

```bash
python eval/grade/grade_new_vl.py --eval-dir results/VL/Qwen3-VL-4B/swap/MathVista-testmini
```

### Omni Audio (`grade_omni_audio.py`)

Extracts final multiple-choice selections while preserving full reasoning traces and per-seed token statistics:

```bash
python eval/grade/grade_omni_audio.py \
  --input results/OMNI_AUDIO/swap/MMAU/mmau_t0.6_p0.95_k20_rp1.0_pp0.0_n4-MNT16384.jsonl \
  --write-details
```
