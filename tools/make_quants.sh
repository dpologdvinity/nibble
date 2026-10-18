#!/usr/bin/env bash
# Builds Q8_0 and Q4_0 test models from the F16 Qwen2.5-0.5B (skips existing outputs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
Q="$ROOT/third_party/build/bin/llama-quantize"
SRC="$ROOT/models/qwen2.5-0.5b-instruct-fp16.gguf"
[ -x "$Q" ] || { echo "missing $Q (run tools/build_llama_cpp.sh)" >&2; exit 1; }
[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
for T in Q8_0 Q4_0; do
  OUT="$ROOT/models/qwen2.5-0.5b-instruct-$(echo "$T" | tr 'A-Z' 'a-z').gguf"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  "$Q" --pure "$SRC" "$OUT" "$T"
done
ls -l "$ROOT"/models/*.gguf
