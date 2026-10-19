#!/usr/bin/env bash
# usage: tools/bench_final.sh [--first nibble|llama.cpp|alternate | --abba <rounds>] <model.gguf> <threads e.g. 1,8> [<model> <threads>]...
# The M6 final-run protocol: for each (model, t), `nibble bench` (pp512, tg128, 5 reps, --mem) and
# tools/bench_llamacpp.sh back to back. --first picks which tool runs first; the default, alternate,
# swaps the order for each successive (model, t). Before every tool run it waits 45 s, then until the
# 1-minute load average is below 2 (giving up after 10 minutes; the row then records the high load).
# Appends to bench/results/m6-final.jsonl and llamacpp-b11715-final.jsonl (env NIBBLE_OUT, LLAMA_OUT);
# progress and gate waits go to stdout. Refuses a -dirty nibble build (env NIBBLE, default build/nibble).
#
# --abba <rounds>: paired-ratio mode. Each (model, t) gets an ABBA block per round (llama.cpp, nibble,
# nibble, llama.cpp; same gate before every run), so slow drift in clock speed or background load
# cancels in the block's llama.cpp/nibble ratio. Rounds are interleaved: every (model, t) gets its
# first block before any gets its second. Both tools append to bench/results/abba.jsonl (env ABBA_OUT);
# each row gets "note":"abba block <id> <A1|B1|B2|A2>", added by this script to the tool's own output.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
nibble=${NIBBLE:-$root/build/nibble}
nibble_out=${NIBBLE_OUT:-$root/bench/results/m6-final.jsonl}
llama_out=${LLAMA_OUT:-$root/bench/results/llamacpp-b11715-final.jsonl}
first=alternate abba=0
abba_out=${ABBA_OUT:-$root/bench/results/abba.jsonl}
if [[ ${1:-} == --first ]]; then first=$2; shift 2; fi
if [[ ${1:-} == --abba ]]; then abba=$2; shift 2; fi
if ! [[ $abba =~ ^[0-9]+$ ]]; then echo "bad --abba: $abba" >&2; exit 2; fi
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
run() {  # run <tool> <model> <t> [note]: with a note, rows go to abba_out with "note" added
  gate
  echo "$(date -u +%FT%TZ) RUN $1 $(basename "$2") t=$3 ${4:-}"
  local out=$nibble_out
  [[ $1 == nibble ]] || out=$llama_out
  [[ -z ${4:-} ]] || out=$(mktemp)
  if [[ $1 == nibble ]]; then
    "$nibble" bench -m "$2" -t "$3" -p 512 -n 128 -r 5 --mem --out "$out"
  else
    P=512 N=128 R=5 "$root/tools/bench_llamacpp.sh" "$2" "$3" "$out"
  fi
  if [[ -n ${4:-} ]]; then
    jq -c --arg note "$4" '. + {note: $note}' "$out" >> "$abba_out"
    rm -f "$out"
  fi
}

if (( abba )); then
  session=$(date -u +%Y%m%dT%H%M%S)
  args=("$@") n=0
  for (( round = 1; round <= abba; ++round )); do
    for (( i = 0; i < ${#args[@]}; i += 2 )); do
      model=${args[i]} threads=${args[i + 1]}
      for t in ${threads//,/ }; do
        n=$((n + 1))
        id=$session-$n
        run llama.cpp "$model" "$t" "abba block $id A1"
        run nibble "$model" "$t" "abba block $id B1"
        run nibble "$model" "$t" "abba block $id B2"
        run llama.cpp "$model" "$t" "abba block $id A2"
      done
    done
  done
  echo "$(date -u +%FT%TZ) DONE"
  exit 0
fi

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
