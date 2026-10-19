#!/usr/bin/env bash
# usage: tools/bench_llamacpp.sh <model> <threads list e.g. 1,2,4> <out.jsonl>
# Runs llama-bench (pp512, tg128, 5 reps; override with P, N, R env) per thread count; appends JSONL in the
# same schema as `nibble bench --out`, with tool "llama.cpp". Needs jq.
set -euo pipefail
model=$1 threads=$2 out=$3
P=${P:-512} N=${N:-128} R=${R:-5}
root=$(cd "$(dirname "$0")/.." && pwd)
tag=$(tr -d '[:space:]' < "$root/tools/llama_cpp_version.txt")
bin=$root/third_party/build/bin/llama-bench
load1=$(cut -d' ' -f1 /proc/loadavg)
for t in ${threads//,/ }; do
  "$bin" -m "$model" -p "$P" -n "$N" -t "$t" -r "$R" -o json | jq -c \
    --arg tag "$tag" --arg model "$(basename "$model")" --argjson t "$t" --argjson load "$load1" '
    .[] | (.samples_ts | sort) as $s | ($s | length) as $n
    | {tool: "llama.cpp", commit: $tag, model: $model,
       test: (if .n_gen > 0 then "tg\(.n_gen)" else "pp\(.n_prompt)" end),
       threads: $t, reps: $n,
       median: (if $n % 2 == 1 then $s[$n/2|floor] else ($s[$n/2-1] + $s[$n/2]) / 2 end),
       min: $s[0], max: $s[-1], unit: "tok/s", load1: $load, date: .test_time}' >> "$out"
done
