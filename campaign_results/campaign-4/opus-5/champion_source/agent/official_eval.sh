#!/usr/bin/env bash
# Official CRPS for every scoring configuration, through agent/predict.py only.
# $1 = agent dir (agent | agent_baseline), $2 = output json, $3 = error log
set -u
DIR="$1"; OUT="$2"; ERR="$3"; END=2026-09-13; DAYS=7; N=1000
: > "$OUT"; : > "$ERR"
emit () { # cfg asset extra-args...
  local CFG="$1" A="$2"; shift 2
  local S
  S=$(uv run "$DIR/predict.py" --asset "$A" --days $DAYS --eval-end $END \
        --num-simulations $N "$@" 2>>"$ERR" | grep "^summary:" | sed 's/^summary: //' | tr "'" '"')
  if [ -z "$S" ]; then echo "FAILED $CFG $A" >> "$ERR"; return; fi
  echo "{\"cfg\":\"$CFG\",\"asset\":\"$A\",\"s\":$S}" >> "$OUT"
  echo "ok $CFG $A" >> "$ERR"
}
for A in BTC ETH SOL XRP HYPE;                                do emit crypto-24h  "$A"; done
for A in XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500;      do emit com-equ-24h "$A"; done
# crypto-1h lives in official_eval_1h.sh: it needs an earlier --eval-end to stay inside the
# snapshot (see suggestions.md A3).
echo "DONE $DIR" >> "$ERR"
