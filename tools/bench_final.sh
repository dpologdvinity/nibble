#!/usr/bin/env bash
# usage: tools/bench_final.sh [--first nibble|llama.cpp|alternate] <model.gguf> <threads e.g. 1,8> [<model> <threads>]...
# The M6 final-run protocol: for each (model, t), `nibble bench` (pp512, tg128, 5 reps, --mem) and
# tools/bench_llamacpp.sh back to back. --first picks which tool runs first; the default, alternate,
# swaps the order for each successive (model, t). Before every tool run it waits 45 s, then until the
# 1-minute load average is below 2 (giving up after 10 minutes; the row then records the high load).
# Appends to bench/results/m6-final.jsonl and llamacpp-b11715-final.jsonl (env NIBBLE_OUT, LLAMA_OUT);
# progress and gate waits go to stdout. Refuses a -dirty nibble build (env NIBBLE, default build/nibble).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
nibble=${NIBBLE:-$root/build/nibble}
nibble_out=${NIBBLE_OUT:-$root/bench/results/m6-final.jsonl}
llama_out=${LLAMA_OUT:-$root/bench/results/llamacpp-b11715-final.jsonl}
first=alternate
if [[ ${1:-} == --first ]]; then first=$2; shift 2; fi
case $first in nibble|llama.cpp|alternate) ;; *) echo "bad --first: $first" >&2; exit 2;; esac
if (( $# == 0 || $# % 2 )); then sed -n 2p "$0" >&2; exit 2; fi
sha=$("$nibble" bench --version)
if [[ $sha == *-dirty ]]; then echo "refusing: $nibble was built from a dirty tree ($sha)" >&2; exit 1; fi

load1() { cut -d' ' -f1 /proc/loadavg; }
gate() {
  local t0=$SECONDS
  sleep 45
  while (( $(echo "$(load1) >= 2" | bc) )) && (( SECONDS - t0 < 600 )); do sleep 10; done
  echo "$(date -u +%FT%TZ) gate waited $((SECONDS - t0))s load1=$(load1)"
}
run() {  # run <tool> <model> <t>
  gate
  echo "$(date -u +%FT%TZ) RUN $1 $(basename "$2") t=$3"
  if [[ $1 == nibble ]]; then
    "$nibble" bench -m "$2" -t "$3" -p 512 -n 128 -r 5 --mem --out "$nibble_out"
  else
    P=512 N=128 R=5 "$root/tools/bench_llamacpp.sh" "$2" "$3" "$llama_out"
  fi
}

k=0
while (( $# )); do
  model=$1 threads=$2
  shift 2
  for t in ${threads//,/ }; do
    lead=$first
    if [[ $first == alternate ]]; then lead=$([[ $((k % 2)) == 0 ]] && echo nibble || echo llama.cpp); fi
    if [[ $lead == nibble ]]; then run nibble "$model" "$t"; run llama.cpp "$model" "$t"
    else run llama.cpp "$model" "$t"; run nibble "$model" "$t"; fi
    k=$((k + 1))
  done
done
echo "$(date -u +%FT%TZ) DONE"
