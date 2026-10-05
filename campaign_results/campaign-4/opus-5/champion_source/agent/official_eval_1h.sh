#!/usr/bin/env bash
# crypto-1h leg only. Separate end date: the backtester pulls minute prices out to
# max(start_time)+26h, which runs past the 2026-09-14 snapshot edge when the 1h prompts
# reach 2026-09-13 -- the day partition does not exist and synth_lib tries to ingest it.
set -u
DIR="$1"; OUT="$2"; ERR="$3"; END=2026-09-12; DAYS=7; N=1000
emit () {
  local CFG="$1" A="$2"; shift 2
  local S
  S=$(uv run "$DIR/predict.py" --asset "$A" --days $DAYS --eval-end $END \
        --num-simulations $N "$@" 2>>"$ERR" | grep "^summary:" | sed 's/^summary: //' | tr "'" '"')
  if [ -z "$S" ]; then echo "FAILED $CFG $A" >> "$ERR"; return; fi
  echo "{\"cfg\":\"$CFG\",\"asset\":\"$A\",\"s\":$S}" >> "$OUT"
  echo "ok $CFG $A" >> "$ERR"
}
for A in BTC ETH SOL XRP HYPE; do emit crypto-1h "$A" --time-increment 60 --time-length 3600; done
echo "DONE-1H $DIR" >> "$ERR"
