import os
import json
import random
import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import re
from typing import List, Dict, Any, Optional, Union, Tuple


# --------------------------------------------------------------------------- #
#                           Answer Extraction                                     #
# --------------------------------------------------------------------------- #

import re
from typing import Optional, List, Tuple

def extract_boxed_content(text: str, start_pos: int) -> Optional[str]:
    """
    Extracts the content inside a \boxed{...} structure starting at start_pos.
    Handles nested braces using a stack.
    """
    if start_pos >= len(text):
        return None
    
    stack = 1
    i = start_pos
    
    while i < len(text) and stack > 0:
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return text[start_pos:i].strip()
        elif text[i] == '\\' and i + 1 < len(text):
            i += 1
        i += 1
    
    return None

def find_all_boxed(text: str) -> List[str]:
    """
    Finds all occurrences of \boxed{content} in the text.
    Returns a list of the contents.
    """
    results = []
    curr = 0
    while True:
        idx = text.find(r'\boxed{', curr)
        if idx == -1: break
        content_start = idx + 7
        res = extract_boxed_content(text, content_start)
        if res is not None:
            results.append(res)
            curr = content_start + len(res)
        else:
            curr = content_start
    return results

def extract_answers(content: str, group_size: int) -> List[Optional[str]]:
    """
    Extract specific number of answers (group_size) from content.
    
    Expected format examples:
    - ### Problem 1 ... Answer1: \boxed{ans}
    - OR just a list of \boxed{ans}
    - OR answers inside <think> blocks (handled by Strategy 0)
    
    Args:
        content: The generated text content
        group_size: The number of answers to extract
        
    Returns:
        List of strings (or None) of length group_size.
    """
    # Initialize list of Nones
    answers: List[Optional[str]] = [None] * group_size

    # --- Strategy 0: Extract from post-</think> content ---
    # Many models put the final concise format after the chain-of-thought.
    think_marker = "</think>"
    if think_marker in content:
        post_think_content = content.split(think_marker)[-1]
        post_boxed = find_all_boxed(post_think_content)
        
        # Map found boxes to answers by index
        for i in range(min(len(post_boxed), group_size)):
            answers[i] = post_boxed[i].strip()
            
        # If we found all expected answers, return early
        if all(a is not None for a in answers):
            return answers
    
    # --- Strategy 1: Extract by "### Problem X" sections ---
    # Split content by problem headers
    problem_sections = re.split(r'###\s*Problem\s*(\d+)', content, flags=re.IGNORECASE)
    
    # re.split returns [preamble, "1", text_of_1, "2", text_of_2, ...]
    for i in range(1, len(problem_sections), 2):
        if i + 1 < len(problem_sections):
            try:
                problem_num = int(problem_sections[i].strip())
            except ValueError:
                continue

            # Only process if this problem number is within our requested group
            if 1 <= problem_num <= group_size:
                section_content = problem_sections[i + 1]
                idx = problem_num - 1 # 0-based index
                
                # If we haven't found this answer yet
                if answers[idx] is None:
                    # 1.1 Look for specific "AnswerX: \boxed{...}" pattern in this section
                    # We use regex to find the label, then use helper for the box to handle nesting
                    label_pattern = re.compile(f"Answer{problem_num}\\s*:\\s*", re.IGNORECASE)
                    label_match = label_pattern.search(section_content)
                    
                    found_specific = False
                    if label_match:
                        # Start searching for \boxed from the end of "AnswerX:"
                        boxed_idx = section_content.find(r'\boxed{', label_match.end())
                        if boxed_idx != -1:
                            extracted = extract_boxed_content(section_content, boxed_idx + 7)
                            if extracted:
                                answers[idx] = extracted
                                found_specific = True
                    
                    # 1.2 Fallback: Find the *first* \boxed{} in this specific Problem section
                    if not found_specific:
                        section_boxes = find_all_boxed(section_content)
                        if section_boxes:
                            answers[idx] = section_boxes[0]

    # --- Strategy 2: Global Search for "Answer N: \boxed{}" ---
    # If specific sections didn't fill everything, look globally for labels
    for i in range(group_size):
        if answers[i] is None:
            problem_num = i + 1
            # Look for "AnswerN:" or "Answer N:" followed immediately by \boxed
            # We find the label position first
            pattern = re.compile(fr"Answer\s*{problem_num}\s*:\s*", re.IGNORECASE)
            match = pattern.search(content)
            
            if match:
                # search for boxed immediately after label
                start_search = match.end()
                boxed_idx = content.find(r'\boxed{', start_search)
                
                # Ensure the boxed is reasonably close to the label (e.g., within 20 chars)
                # to avoid false positives if the label is mentioned far from the box
                if boxed_idx != -1 and (boxed_idx - start_search) < 50:
                    extracted = extract_boxed_content(content, boxed_idx + 7)
                    if extracted:
                        answers[i] = extracted

    # --- Strategy 3: Final Fallback - Sequential Boxed ---
    # If we still have missing answers, just grab all boxed items in the entire text
    # and fill the missing slots in order.
    if any(a is None for a in answers):
        all_boxed = find_all_boxed(content)
        boxed_idx = 0
        
        for i in range(group_size):
            if answers[i] is None:
                # Try to find the next available boxed item
                if boxed_idx < len(all_boxed):
                    answers[i] = all_boxed[boxed_idx]
                    boxed_idx += 1
            else:
                # If answer[i] is already filled, we assume it corresponds to 
                # one of the items in all_boxed. To prevent "stealing" a box 
                # that belongs to the next missing answer, we should increment boxed_idx
                # if the current filled answer matches the current boxed item.
                if boxed_idx < len(all_boxed) and answers[i] == all_boxed[boxed_idx]:
                    boxed_idx += 1

    return answers


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
    
    # group size
    parser.add_argument("--group_size", type=int, default=3)

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

def sys_prompt_parser(group_size: int) -> str:
    
    example_answers = "\n".join([
        f"            After Problem {i}: Answer{i}: \\boxed{{...}}" 
        for i in range(1, group_size + 1)
    ])

    base_prompt = f"""
    You are an expert mathematics tutor. 
    Your task is to solve **{group_size}** math problems sequentially in a single response.
    Please strictly follow these rules:
        1. Use Markdown headers (### Problem X) to separate each problem.
        2. For each problem, show detailed step-by-step reasoning, then immediately put the final answer right after the reasoning.
        3. Put the final answer for each problem immediately after solving it, in the following format:
{example_answers}
        4. Each answer should appear right after its corresponding problem's reasoning, before moving to the next problem.
        5. Do not include any other text in your response.
    """.strip()

    return base_prompt

# --------------------------------------------------------------------------- #
#                               Worker                                          #
# --------------------------------------------------------------------------- #
def worker_process(args):
    samples, seeds, gpu_id, model_name, temp, top_p, max_tokens, group_size = args

    sys_prompt_batch = sys_prompt_parser(group_size)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name, enforce_eager=True, max_model_len=32768, gpu_memory_utilization=0.6)
    
    results = []
    for seed in seeds:
        sampling = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        
        batchs = [samples[i:i + group_size] for i in range(0, len(samples), group_size)]
        messages = []
        for batch in batchs:
            user_content_batch = f"Please solve the following {group_size} problems:\n\n" + \
                        "\n".join([f"Problem {i+1}:\n{s['prompt']}" for i, s in enumerate(batch)])
            messages.append([{"role": "system", "content": sys_prompt_batch}, {"role": "user", "content": user_content_batch}])
            
        print(messages[0])
        outputs = llm.chat(messages, sampling, use_tqdm=False)

        for batch, out in zip(batchs, outputs):
            generated_tokens = out.outputs[0].token_ids
            token_length = len(generated_tokens)
            response = out.outputs[0].text
            answers = extract_answers(response, group_size)
            for i, s in enumerate(batch):
                results.append(
                    {
                        "example_id": s["example_id"],
                        "prompt": s["prompt"],
                        "answer": s["answer"],
                        "seed": seed,
                        "response": answers[i],
                        "token_length": token_length / group_size,  # average per problem
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

    model_short = args.model.split("/")[-1] + f"exam_{args.group_size}"
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
             args.model, args.temperature, args.top_p, args.max_tokens, args.group_size)
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

    