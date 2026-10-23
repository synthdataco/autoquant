#!/bin/bash
# usage: agent/sweep.sh START END COMP "tag1:K=V,K=V" "tag2:..." ; runs 3 at a time, prints per-asset tables
S=$1; E=$2; C=$3; shift 3
run() { spec=$1; tag=${spec%%:*}; envs=${spec#*:}; envs=${envs//,/ }
  uv run agent/evalrun.py --comp $C --start $S --end $E --tag $tag --jobs 5 --env $envs 2>&1 | grep -A20 "^asset" > agent/results/log_${tag}_${C}.txt; echo "== $tag $C"; cat agent/results/log_${tag}_${C}.txt; }
mkdir -p agent/results
N=0; for spec in "$@"; do run "$spec" & N=$((N+1)); if [ $N -ge 3 ]; then wait -n; N=$((N-1)); fi; done; wait
