import os
import json
import random
import argparse
import traceback
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


TASK_REGISTRY = {
    "AIME24": ("AIME24/test.parquet", 1),
    "AIME25": ("AIME25/test.parquet", 1),
    "AMC23": ("AMC23/test.parquet", 1),
    "MATH-500": ("MATH-500/test.parquet", 1),
    "Minerva": ("Minerva/test.parquet", 1),
    "Olympiad-Bench": ("Olympiad-Bench/test.parquet", 1),
    "BRUMO25": ("BRUMO25/test.parquet", 1),
    "CMIMC25": ("CMIMC25/test.parquet", 1),
    "HMMT25": ("HMMT25/test.parquet", 1),
}


def parse_args():
    parser = argparse.ArgumentParser("Evaluation runner (supports LoRA)")

    # model / decoding
    parser.add_argument("--model", type=str, required=True,
                        help="Base model path/repo, or LoRA adapter path (auto-detected).")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path/repo when --model points to a LoRA adapter.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter checkpoint directory (PEFT format).")
    parser.add_argument("--lora_name", type=str, default="adapter")
    parser.add_argument("--max_lora_rank", type=int, default=None,
                        help="Override LoRA rank for vLLM. Defaults to adapter_config.json['r'] if available.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Override subdirectory name under --out_dir.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="number of samples per problem (override task default)")

    # tasks
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--tasks", nargs="+", required=True,
                        choices=TASK_REGISTRY.keys())

    # hardware
    parser.add_argument("--gpus", type=str, default="0",
                        help="comma separated gpu ids, e.g. 0,1,2,3")

    # output
    parser.add_argument("--out_dir", type=str, default="eval_outputs")

    return parser.parse_args()


def load_samples(filepath: str):
    df = pd.read_parquet(filepath)
    if any(k in filepath for k in ["BRUMO25", "CMIMC25", "HMMT25"]):
        return [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    return [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"].strip(),
            "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
        }
        for i in range(len(df))
    ]


def split_seeds(seeds, num_workers):
    chunks = [[] for _ in range(num_workers)]
    for i, s in enumerate(seeds):
        chunks[i % num_workers].append(s)
    return chunks


def _read_adapter_config(adapter_dir: Path):
    cfg_file = adapter_dir / "adapter_config.json"
    if not cfg_file.exists():
        return None
    with cfg_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_lora_adapter(path_str: str) -> bool:
    p = Path(path_str)
    return p.is_dir() and (p / "adapter_config.json").exists() and not (p / "config.json").exists()


def resolve_model_config(args):
    # Compatible mode:
    # 1) --model=<full model>, optional --lora_path=<adapter>
    # 2) --model=<adapter>, auto-detect LoRA + infer base model from adapter_config.json
    model_input = args.model
    lora_path = args.lora_path
    base_model = args.base_model

    if lora_path is None and _looks_like_lora_adapter(model_input):
        lora_path = model_input

    adapter_cfg = None
    if lora_path is not None:
        adapter_cfg = _read_adapter_config(Path(lora_path))
        if base_model is None and adapter_cfg is not None:
            base_model = adapter_cfg.get("base_model_name_or_path")
        if base_model is None:
            raise ValueError(
                "LoRA adapter detected/provided but base model is unknown. "
                "Pass --base_model or ensure adapter_config.json contains base_model_name_or_path."
            )
    else:
        base_model = model_input

    if base_model is None:
        raise ValueError("Unable to resolve base model.")

    max_lora_rank = args.max_lora_rank
    if lora_path is not None and max_lora_rank is None and adapter_cfg is not None:
        rank = adapter_cfg.get("r")
        if isinstance(rank, int):
            max_lora_rank = rank

    if args.out_name:
        out_name = args.out_name
    elif lora_path is not None:
        lp = Path(lora_path)
        out_name = f"{lp.parent.name}_{lp.name}exam_1"
    else:
        out_name = Path(base_model).name + "exam_1"

    return {
        "base_model": base_model,
        "lora_path": lora_path,
        "lora_name": args.lora_name,
        "max_lora_rank": max_lora_rank or 64,
        "out_name": out_name,
    }


def worker_process(args):
    sys_prompt_single = """You are an expert mathematics tutor.
    Your task is to solve the **single** math problem provided below.
    Reason step-by-step and provide a clear conclusion.
    Put the final answer in \\boxed{}."""

    (
        samples,
        seeds,
        gpu_id,
        base_model,
        lora_path,
        lora_name,
        max_lora_rank,
        temp,
        top_p,
        max_tokens,
        gpu_memory_utilization,
    ) = args

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        llm_kwargs = dict(
            model=base_model,
            enforce_eager=True,
            max_model_len=32768,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if lora_path is not None:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank

        llm = LLM(**llm_kwargs)
        lora_request = None
        if lora_path is not None:
            lora_request = LoRARequest(lora_name, 1, lora_path)

        results = []
        for seed in seeds:
            sampling = SamplingParams(
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
            messages = [[
                {"role": "system", "content": sys_prompt_single},
                {"role": "user", "content": "Please solve this problem:" + s["prompt"]},
            ] for s in samples]

            chat_kwargs = dict(messages=messages, sampling_params=sampling, use_tqdm=False)
            if lora_request is not None:
                chat_kwargs["lora_request"] = lora_request
            outputs = llm.chat(**chat_kwargs)

            for s, out in zip(samples, outputs):
                generated_tokens = out.outputs[0].token_ids
                token_length = len(generated_tokens)
                results.append(
                    {
                        "example_id": s["example_id"],
                        "prompt": s["prompt"],
                        "answer": s["answer"],
                        "seed": seed,
                        "response": out.outputs[0].text,
                        "token_length": token_length,
                    }
                )
        return results
    except Exception as e:
        # Convert to a pickle-friendly exception for ProcessPoolExecutor.
        msg = (
            f"Worker failed on GPU {gpu_id}\n"
            f"base_model={base_model}\n"
            f"lora_path={lora_path}\n"
            f"{type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        raise RuntimeError(msg) from None


def main():
    args = parse_args()
    resolved = resolve_model_config(args)

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_workers = len(gpu_ids)

    out_root = Path(args.out_dir) / resolved["out_name"]
    out_root.mkdir(parents=True, exist_ok=True)

    print("Resolved model config:")
    print(f"  base_model: {resolved['base_model']}")
    print(f"  lora_path: {resolved['lora_path']}")
    print(f"  lora_name: {resolved['lora_name']}")
    print(f"  max_lora_rank: {resolved['max_lora_rank']}")
    print(f"  out_root: {out_root}")

    for task_name in args.tasks:
        rel_path, default_n = TASK_REGISTRY[task_name]
        N = args.batch_size or default_n
        task_path = f"{args.data_dir}/{rel_path}"

        print(f"\n==> Task {task_name} | N={N}")
        samples = load_samples(task_path)

        seeds = random.sample(range(2**31 - 1), N)
        seed_chunks = split_seeds(seeds, num_workers)

        args_list = [
            (
                samples,
                seed_chunks[i],
                gpu_ids[i],
                resolved["base_model"],
                resolved["lora_path"],
                resolved["lora_name"],
                resolved["max_lora_rank"],
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.gpu_memory_utilization,
            )
            for i in range(num_workers)
        ]

        all_results = []
        with concurrent.futures.ProcessPoolExecutor(num_workers) as ex:
            futures = [ex.submit(worker_process, a) for a in args_list]
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc=f"{task_name}"):
                all_results.extend(f.result())

        out_file = out_root / (
            f"{task_name.lower()}_t{args.temperature}_p{args.top_p}"
            f"_n{N}-MNT{args.max_tokens}.jsonl"
        )
        with out_file.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
