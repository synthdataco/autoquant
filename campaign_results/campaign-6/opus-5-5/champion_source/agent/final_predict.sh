#!/bin/bash
# Official per-asset backtests (agent/predict.py, 1000 paths) for the champion and the starter, all configs.
# usage: agent/final_predict.sh TAG [ENV=...]   (model = agent/modeling.py; TAG=starter swaps in the starter)
cd /workspace; TAG=$1; shift
mkdir -p agent/results/predict_$TAG
run() { # asset days end ti tl
  P=agent; [ "$TAG" = starter ] && P=starter
  env "${@:6}" uv run $P/predict.py --asset $1 --days $2 --eval-end $3 --time-increment $4 --time-length $5 --latest-formula \
    > agent/results/predict_$TAG/$1_$5.txt 2>&1; }
export -f run; export TAG
{ for a in BTC ETH SOL XRP HYPE XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500; do echo "$a 15 2026-10-11 300 86400 $*"; done
  for a in BTC ETH SOL XRP HYPE; do echo "$a 19 2026-10-11 60 3600 $*"; done; } | xargs -P 9 -L 1 bash -c 'run $@' _
grep -H "summary" agent/results/predict_$TAG/*.txt | sed -E "s/.*predict_$TAG\/([A-Z0-9]+)_([0-9]+).txt.*'num_prompts': ([0-9]+), 'mean_crps': ([0-9.]+).*/\1 \2 \3 \4/"
