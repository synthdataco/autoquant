#!/bin/bash
# Official numbers: agent/predict.py (synth-lib backtest, 1000 paths, validator prompt times),
# one run per (configuration, asset), 10 days of scored prompts before EVAL_END.
# usage: official_eval.sh <predict.py path> <out dir> [eval_end] [days]
PRED=$1; OUT=$2; END=${3:-2026-11-08}; DAYS=${4:-10}
mkdir -p $OUT
cd /workspace
run() { uv run $PRED --asset $1 --days $DAYS --eval-end $END $2 > $OUT/$3_$1.log 2>&1; }
for a in BTC ETH SOL XRP HYPE; do run $a "" crypto-24h & done; wait
for a in XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500; do run $a "" com-equ-24h & done; wait
for a in BTC ETH SOL XRP HYPE; do run $a "--time-length 3600 --time-increment 60" crypto-1h & done; wait
