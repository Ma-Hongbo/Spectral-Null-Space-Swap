import os
import json
import random
import argparse
import tempfile
import shutil
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


def prepare_tokenizer_for_vllm(model_path: str) -> str:
    model_dir = Path(model_path)
    tok_cfg = model_dir / "tokenizer_config.json"
    if not tok_cfg.exists():
        return model_path

    try:
        with tok_cfg.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return model_path

    if cfg.get("tokenizer_class") != "TokenizersBackend":
        return model_path

    tmp_dir = Path(tempfile.mkdtemp(prefix="vllm_tokenizer_fix_"))
    files_to_copy = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
    ]
    for name in files_to_copy:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, tmp_dir / name)

    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    with (tmp_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Using patched tokenizer metadata at: {tmp_dir}")
    return str(tmp_dir)


def resolve_chat_template_path(model_path: str, tokenizer_path: str) -> str | None:
    candidates = [
        Path(tokenizer_path) / "chat_template.jinja",
        Path(tokenizer_path) / "chat_template.json",
        Path(model_path) / "chat_template.jinja",
        Path(model_path) / "chat_template.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


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
        tokenizer_name,
        chat_template_path,
        temp,
        top_p,
        max_tokens,
    ) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm_kwargs = dict(
        model=model_name,
        tokenizer=tokenizer_name,
        enforce_eager=True,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
    )
    if chat_template_path is not None:
        llm_kwargs["chat_template"] = chat_template_path
    llm = LLM(**llm_kwargs)
    
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
    tokenizer_name = prepare_tokenizer_for_vllm(args.model)
    chat_template_path = resolve_chat_template_path(args.model, tokenizer_name)

    for task_name in args.tasks:
        rel_path, default_n = TASK_REGISTRY[task_name]
        N = args.batch_size or default_n
        task_path = f"{args.data_dir}/{rel_path}"

        print(f"\n==> Task {task_name} | N={N}")

        samples = load_samples(task_path)

        seeds = random.sample(range(2**31 - 1), N)
        seed_chunks = split_seeds(seeds, num_workers)

        args_list = [
            (samples, seed_chunks[i], gpu_ids[i], args.model, tokenizer_name, chat_template_path,
             args.temperature, args.top_p, args.max_tokens)
            for i in range(num_workers)
        ]

        all_results = []
        if num_workers == 1:
            # Avoid multiprocessing semaphore requirements in restricted envs.
            all_results.extend(worker_process(args_list[0]))
        else:
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
