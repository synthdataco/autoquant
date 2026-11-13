#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# wait for fB3 training, then official evals; compact output
while [ ! -f scratch/train_fB3.log ] || [ $(grep -c "ntr=" scratch/train_fB3.log) -lt 18 ]; do sleep 20; done
export CORE_MATCH=0
ev() { # name params days end tl
  CORE_PARAMS_DIR=$2 uv run agent/predict.py --assets all --days $3 --eval-end $4 --time-length $5 --jobs 13 --out scratch/runs/$1.parquet 2>&1 | grep -v -i warn > scratch/runs/$1.log
  echo "## $1"; grep -E "AGG" scratch/runs/$1.log | cut -c1-140; awk 'NR>2 && $1!="AGG" && $1!="per-prompt" {printf "%s %s %s %s %s | ", $1, $3, $4, $5, $6}' scratch/runs/$1.log; echo
}
ev v3fB_24h scratch/params_fB3 36 2026-11-07 86400
ev v3fB_1h scratch/params_fB3 36 2026-11-07 3600
# one-clock profile with the same fitted parameters, on the post-DST week only (London switched 10-25, NY 11-01)
mkdir -p scratch/params_fB3_1c; cp scratch/params_fB3/model_*.json scratch/params_fB3_1c/
ev v3fB_2c_dst_24h scratch/params_fB3 12 2026-11-07 86400
ev v3fB_1c_dst_24h scratch/params_fB3_1c 12 2026-11-07 86400
ev v3fB_2c_dst_1h scratch/params_fB3 12 2026-11-07 3600
ev v3fB_1c_dst_1h scratch/params_fB3_1c 12 2026-11-07 3600
echo ALLDONE
