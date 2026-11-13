#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# usage: research/eval_all.sh <params_dir> <tag> [days] [eval_end] [modeling]
# Official evaluation (agent/predict.py, 1000 paths, validator prompts) of the three configurations.
P=$1; TAG=$2; DAYS=${3:-100}; END=${4:-2026-11-07}; MOD=${5:-agent/modeling.py}
mkdir -p scratch/runs
CORE_PARAMS_DIR=$P uv run agent/predict.py --modeling $MOD --assets all --days $DAYS --eval-end $END --jobs 13 --out scratch/runs/${TAG}_24h.parquet 2>&1 | grep -v -i warn > scratch/runs/${TAG}_24h.log
CORE_PARAMS_DIR=$P uv run agent/predict.py --modeling $MOD --assets all --time-length 3600 --days $DAYS --eval-end $END --jobs 13 --out scratch/runs/${TAG}_1h.parquet 2>&1 | grep -v -i warn > scratch/runs/${TAG}_1h.log
cat scratch/runs/${TAG}_24h.log scratch/runs/${TAG}_1h.log
