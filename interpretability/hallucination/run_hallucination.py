#!/usr/bin/env python3
"""Hallucination evaluation orchestrator — dispatches TruthfulQA & SimpleQA
jobs across GPUs. Adapted from run_ablation.py.

Usage (on server):
  python run_hallucination.py [--benchmarks truthfulqa simpleqa] [--limit N]
"""
import os, sys, time, json, subprocess, argparse

# Machine-specific locations come from the environment (see README):
#   S3_ROOT      directory holding paths2507.env, data/ and models/   (default: .)
#   S3_HF_HOME   HuggingFace cache, if not the usual ~/.cache/huggingface
ROOT = os.environ.get("S3_ROOT", ".")
if os.environ.get("S3_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["S3_HF_HOME"])
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

WORK_GPUS = [0, 3, 4, 7]  # pick free GPUs; adjust based on nvidia-smi
MINFREE = 20000           # less memory needed than math eval (shorter context)
TOTAL_MIB = 46068
MAX_MODEL_LEN = 16384
MAX_TOKENS_TQA = 16000
MAX_TOKENS_SQA = 8192
KV_CACHE_DTYPE = "fp8"
ROOT = ROOT
OUT_DIR = os.path.join(ROOT, "analysis/hallucination_eval")

BENCHMARKS = {
    "truthfulqa": {
        "script": "eval_truthfulqa.py",
        "data": "data/truthfulqa_mc.jsonl",
        "max_tokens": MAX_TOKENS_TQA,
    },
    "simpleqa": {
        "script": "eval_simpleqa.py",
        "data": "data/simpleqa.jsonl",
        "max_tokens": MAX_TOKENS_SQA,
    },
}


def paths_env():
    env = {}
    for line in open(os.path.join(ROOT, "paths2507.env")):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def gpu_free():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"]).decode()
    d = {}
    for line in out.strip().splitlines():
        i, f = line.split(",")
        d[int(i)] = int(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pe = paths_env()
    MODELS = {
        "base":     pe["BASE2"],
        "subonly":   "models/subonly_2507",
        "perponly":  "models/perponly_2507",
        "full":      pe["RL2"],
    }
    ORDER = ["base", "subonly", "perponly", "full"]
    os.makedirs(OUT_DIR, exist_ok=True)

    work = []
    for bench in args.benchmarks:
        if bench not in BENCHMARKS:
            print(f"Unknown benchmark: {bench}")
            sys.exit(1)
        for model_name in ORDER:
            outf = os.path.join(OUT_DIR, f"{bench}_{model_name}.json")
            if not os.path.exists(outf):
                work.append((bench, model_name))
            else:
                print(f"SKIP (exists): {bench}/{model_name}")
    print(f"{time.strftime('%T')} work items: {len(work)}", flush=True)

    if not work:
        print("Nothing to do.")
        return

    base_env = dict(os.environ)
    base_env.update({
        "HF_HOME": HF_CACHE,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    running = {}
    held = set()
    fails = {}

    while work or running:
        for pid, r in list(running.items()):
            if r["proc"].poll() is not None:
                r["logf"].close()
                held.discard(r["gpu"])
                del running[pid]
                outf = os.path.join(OUT_DIR,
                                    f"{r['bench']}_{r['model']}.json")
                if os.path.exists(outf):
                    print(f"{time.strftime('%T')} DONE {r['bench']}/{r['model']} "
                          f"(gpu{r['gpu']})", flush=True)
                else:
                    k = (r["bench"], r["model"])
                    fails[k] = fails.get(k, 0) + 1
                    print(f"{time.strftime('%T')} FAIL {r['bench']}/{r['model']} "
                          f"(gpu{r['gpu']}) attempt {fails[k]}", flush=True)
                    if fails[k] <= 3:
                        work.append(k)

        if work:
            free = gpu_free()
            for g in WORK_GPUS:
                if not work:
                    break
                if g in held:
                    continue
                if free.get(g, 0) >= MINFREE:
                    bench, model_name = work.pop(0)
                    frac = min(0.85, (free[g] - 2000) / TOTAL_MIB)
                    outf = os.path.join(OUT_DIR, f"{bench}_{model_name}.json")
                    logf_path = os.path.join(OUT_DIR, f"{bench}_{model_name}.log")
                    bconf = BENCHMARKS[bench]

                    env = dict(base_env)
                    env["CUDA_VISIBLE_DEVICES"] = str(g)
                    env["VLLM_CACHE_ROOT"] = f"/tmp/vllm_hall_{bench}_{model_name}"
                    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/ti_hall_{bench}_{model_name}"
                    env["TRITON_CACHE_DIR"] = f"/tmp/tr_hall_{bench}_{model_name}"

                    cmd = [
                        ".venv/bin/python", bconf["script"],
                        "--model", MODELS[model_name],
                        "--data", bconf["data"],
                        "--out", outf,
                        "--max-tokens", str(bconf["max_tokens"]),
                        "--max-model-len", str(MAX_MODEL_LEN),
                        "--kv-cache-dtype", KV_CACHE_DTYPE,
                        "--gpu-frac", f"{frac:.3f}",
                    ]
                    if args.limit:
                        cmd.extend(["--limit", str(args.limit)])

                    logf = open(logf_path, "w")
                    proc = subprocess.Popen(
                        cmd, cwd=ROOT, env=env,
                        stdout=logf, stderr=subprocess.STDOUT)
                    running[proc.pid] = dict(
                        gpu=g, bench=bench, model=model_name,
                        proc=proc, logf=logf)
                    held.add(g)
                    print(f"{time.strftime('%T')} LAUNCH {bench}/{model_name} -> "
                          f"gpu{g} frac={frac:.3f}", flush=True)
                    time.sleep(8)

        time.sleep(20)

    print(f"\n{time.strftime('%T')} ALL DONE", flush=True)

    # Print summary
    print("\n" + "=" * 60)
    for bench in args.benchmarks:
        print(f"\n--- {bench.upper()} ---")
        for m in ORDER:
            outf = os.path.join(OUT_DIR, f"{bench}_{m}.json")
            if os.path.exists(outf):
                data = json.load(open(outf))
                if bench == "truthfulqa":
                    print(f"  {m:10s}: MC1 acc = {data['mc1_accuracy']:.4f} "
                          f"({data['n_correct']}/{data['n_total']})")
                elif bench == "simpleqa":
                    print(f"  {m:10s}: acc = {data['accuracy']:.4f} "
                          f"(C={data['counts']['CORRECT']}/"
                          f"I={data['counts']['INCORRECT']}/"
                          f"NA={data['counts']['NOT_ATTEMPTED']})")


if __name__ == "__main__":
    main()
