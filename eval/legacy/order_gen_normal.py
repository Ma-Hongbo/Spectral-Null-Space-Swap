import os
import json
import random
import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

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


# --------------------------------------------------------------------------- #
#                               Worker                                          #
# --------------------------------------------------------------------------- #
def worker_process(args):
    
    sys_prompt_single = """You are an expert mathematics tutor.
    Your task is to solve the **single** math problem provided below.
    Reason step-by-step and provide a clear conclusion.
    Put the final answer in \boxed{}."""
    
    samples, seeds, gpu_id, model_name, temp, top_p, max_tokens = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name, enforce_eager=True, max_model_len=32768, gpu_memory_utilization=0.7)
    
    results = []
    for seed in seeds:
        sampling = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        messages = [[{"role": "system", "content": sys_prompt_single}, {"role": "user", "content": "Please solve this problem:" + s["prompt"]}] for s in samples]
        outputs = llm.chat(messages, sampling, use_tqdm=False)

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


# --------------------------------------------------------------------------- #
#                                   Main                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_workers = len(gpu_ids)

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
            (samples, seed_chunks[i], gpu_ids[i],
             args.model, args.temperature, args.top_p, args.max_tokens)
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