#!/bin/bash
# After the fB3 training: retrain a subset of (asset, format) with one v3 component switched off each time.
export OMP_NUM_THREADS=1
while [ ! -f scratch/train_fB3.log ] || [ $(grep -c "ntr=" scratch/train_fB3.log) -lt 18 ]; do sleep 20; done
A24=BTC,ETH,SP500,XAU,WTIOIL,AAPLX; A1=BTC,SOL
run() { # name, extra args
  n=$1; shift
  uv run python research/train.py --ds scratch/ds_fB3 --out scratch/params_fB3_abl --train-end 2026-10-01 --val-start 2026-10-01 --steps 1200 --assets $A24 --tags 24h --no-save "$@" > scratch/abl_$n.log 2>&1
  uv run python research/train.py --ds scratch/ds_fB3 --out scratch/params_fB3_abl --train-end 2026-10-01 --val-start 2026-10-01 --steps 1200 --assets $A1 --tags 1h --no-save "$@" >> scratch/abl_$n.log 2>&1
}
run geo --freeze p=0.02
run nosess --freeze gamma=0 --mask-fam sess
run noextra --freeze ce=0
run v2like --freeze p=0.02,gamma=0,ce=0 --mask-fam sess
echo ABLDONE >> scratch/abl_v2like.log
