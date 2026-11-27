import os
import json
import random
import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
#                           Task registry                                      #
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#                              Argument parser                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser("Evaluation runner")

    # model / decoding
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="number of samples per problem (override task default)")
    parser.add_argument("--micro_batch_size", type=int, default=200,
                        help="number of prompts per HF generate call (smaller -> lower VRAM)")
    parser.add_argument("--max_context_tokens", type=int, default=32768,
                        help="tokenizer truncation length for prompt context")

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


# --------------------------------------------------------------------------- #
#                               Helpers                                         #
# --------------------------------------------------------------------------- #
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
    else:
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


def parse_gpu_ids(gpus_arg: str):
    text = str(gpus_arg).strip()
    if not text:
        raise ValueError("--gpus is empty.")

    # List mode: "0,1,2,3"
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    # Single id mode: "8" -> [8]
    n = int(text)
    if n < 0:
        raise ValueError("--gpus must be non-negative.")
    return [n]


def iter_chunks(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


# --------------------------------------------------------------------------- #
#                               Worker                                          #
# --------------------------------------------------------------------------- #
def worker_process(args):
    
    sys_prompt_single = """You are an expert mathematics tutor.
    Your task is to solve the **single** math problem provided below.
    Reason step-by-step and provide a clear conclusion.
    Put the final answer in \boxed{}."""
    
    (
        samples,
        seeds,
        gpu_id,
        model_name,
        temp,
        top_p,
        max_tokens,
        micro_batch_size,
        max_context_tokens,
    ) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for sample_chunk in iter_chunks(samples, max(1, micro_batch_size)):
            messages = [
                [
                    {"role": "system", "content": sys_prompt_single},
                    {"role": "user", "content": "Please solve this problem:" + s["prompt"]},
                ]
                for s in sample_chunk
            ]

            prompts = []
            for m in messages:
                try:
                    prompt = tokenizer.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = (
                        f"System: {sys_prompt_single}\n\n"
                        f"User: {m[1]['content']}\n\nAssistant:"
                    )
                prompts.append(prompt)

            model_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_context_tokens,
            ).to(model.device)

            input_len = model_inputs["input_ids"].shape[1]
            available_new_tokens = max(1, max_context_tokens - input_len)
            safe_max_new_tokens = min(max_tokens, available_new_tokens)

            try:
                with torch.no_grad():
                    generated = model.generate(
                        **model_inputs,
                        max_new_tokens=safe_max_new_tokens,
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Fallback for very long cases: decode one-by-one.
                generated_rows = []
                for i in range(len(sample_chunk)):
                    single_inputs = {
                        k: v[i:i + 1] for k, v in model_inputs.items()
                    }
                    single_input_len = single_inputs["input_ids"].shape[1]
                    single_available = max(1, max_context_tokens - single_input_len)
                    single_safe_max_new = min(max_tokens, single_available)
                    with torch.no_grad():
                        row = model.generate(
                            **single_inputs,
                            max_new_tokens=single_safe_max_new,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    generated_rows.append(row)
                generated = torch.cat(generated_rows, dim=0)

            output_ids = generated[:, model_inputs["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for s, response in zip(sample_chunk, responses):
                token_length = len(tokenizer.encode(response, add_special_tokens=False))
                results.append(
                    {
                        "example_id": s["example_id"],
                        "prompt": s["prompt"],
                        "answer": s["answer"],
                        "seed": seed,
                        "response": response,
                        "token_length": token_length,
                    }
                )
    return results


# --------------------------------------------------------------------------- #
#                                   Main                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    num_workers = len(gpu_ids)
    print(f"Using GPU ids: {gpu_ids}")

    model_short = args.model.split("/")[-1] + "exam_1"
    out_root = Path(args.out_dir) / model_short
    out_root.mkdir(parents=True, exist_ok=True)

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
                args.model,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.micro_batch_size,
                args.max_context_tokens,
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

        print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
