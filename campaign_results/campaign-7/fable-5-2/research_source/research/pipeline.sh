#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# usage: research/pipeline.sh <name> <fit_end> <train_end> <val_start> [extra train args]
# profile fit -> dataset -> training, all into scratch/params_<name>, scratch/ds_<name>
NAME=$1; FIT_END=$2; TRAIN_END=$3; VAL_START=$4; shift 4
set -e
uv run python research/fit_profiles.py --fit-end $FIT_END --out scratch/params_$NAME > scratch/fit_$NAME.log 2>&1
uv run python research/build_dataset.py --params scratch/params_$NAME --out scratch/ds_$NAME > scratch/build_$NAME.log 2>&1
uv run python research/train.py --ds scratch/ds_$NAME --out scratch/params_$NAME --train-end $TRAIN_END --val-start $VAL_START "$@" > scratch/train_$NAME.log 2>&1
echo done $NAME
