#!/bin/bash
# Launch one grad_cos_7datasets.py per dataset, one GPU each (the 7 datasets are
# independent, so they shard trivially). GPUs are chosen by free memory at launch time;
# if fewer qualify than there are datasets, the extra datasets queue behind the others
# on the chosen cards rather than oversubscribing a busy one.
#
#   bash run_7datasets.sh            # all 7
#   MINFREE=30 bash run_7datasets.sh # relax the free-memory bar
#
# Each dataset writes its own JSON under $SAR/grad_cos_7datasets/ and its own log under
# $SAR/logs/grad_cos_7d/, so a failure on one card cannot damage another's result.
set -uo pipefail

SAR="${S3_ROOT:?set S3_ROOT to the directory holding paths2507.env, data/ and models/}"
PY=$SAR/.venv/bin/python
SCRIPT=$SAR/grad_cos_7datasets.py
LOGDIR=$SAR/logs/grad_cos_7d
DATASETS="${DATASETS:-aime24 aime25 amc23 cmimc25 hmmt25 math500 gsm8k}"
MINFREE=${MINFREE:-34}          # GB free required on a card before we use it

mkdir -p "$LOGDIR" "$SAR/grad_cos_7datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
[ -n "${S3_HF_HOME:-}" ] && export HF_HOME="$S3_HF_HOME" || true

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F', ' -v m=$((MINFREE * 1024)) '$2 >= m {print $2, $1}' | sort -rn | awk '{print $2}')
if [ ${#GPUS[@]} -eq 0 ]; then
  echo "no GPU has >= ${MINFREE}GB free; lower MINFREE or wait" >&2
  exit 1
fi
echo "candidate GPUs (most free first): ${GPUS[*]}"

# deal datasets onto the candidate cards
declare -A QUEUE
i=0
for ds in $DATASETS; do
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  QUEUE[$gpu]="${QUEUE[$gpu]:-} $ds"
  i=$(( i + 1 ))
done

pids=()
for gpu in "${!QUEUE[@]}"; do
  dslist=${QUEUE[$gpu]}
  echo "GPU $gpu -> $dslist"
  (
    for ds in $dslist; do
      echo "=== $ds on GPU $gpu  $(date +%H:%M:%S) ==="
      CUDA_VISIBLE_DEVICES=$gpu $PY "$SCRIPT" --datasets "$ds" >"$LOGDIR/$ds.log" 2>&1
      echo "=== $ds exit=$? $(date +%H:%M:%S) ==="
    done
  ) >"$LOGDIR/gpu$gpu.driver.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} GPU workers; waiting"
wait
echo "all workers finished; logs in $LOGDIR"
$PY "$SCRIPT" --report
