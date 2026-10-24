#!/usr/bin/env bash
# Builds the quantized test models (skips missing sources and existing outputs):
#   - pure Q8_0 and Q4_0 of the F16 Qwen2.5-0.5B (--pure: every tensor the same type);
#   - Q4_K_M, Q5_K_M and Q6_K of each F16 model, as llama-quantize's default mix (no --pure), which is
#     what real k-quant files contain: Q5_0/Q5_1/Q8_0 fallbacks next to the K types.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
Q="$ROOT/third_party/build/bin/llama-quantize"
SRC="$ROOT/models/qwen2.5-0.5b-instruct-fp16.gguf"
[ -x "$Q" ] || { echo "missing $Q (run tools/build_llama_cpp.sh)" >&2; exit 1; }
[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
for T in Q8_0 Q4_0; do
  OUT="$ROOT/models/qwen2.5-0.5b-instruct-$(lower "$T").gguf"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  "$Q" --pure "$SRC" "$OUT.tmp" "$T"
  mv "$OUT.tmp" "$OUT"
done
for PAIR in qwen2.5-0.5b-instruct:qwen2.5-0.5b-instruct-fp16 llama-3.2-1b-instruct:llama-3.2-1b-instruct-f16 \
            tinyllama-1.1b-chat:tinyllama-1.1b-chat-f16; do
  BASE="${PAIR%%:*}"; F16="$ROOT/models/${PAIR##*:}.gguf"
  [ -f "$F16" ] || { echo "skip $BASE: missing $F16" >&2; continue; }
  for T in Q4_K_M Q5_K_M Q6_K; do
    OUT="$ROOT/models/$BASE-$(lower "$T").gguf"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    "$Q" "$F16" "$OUT.tmp" "$T"
    mv "$OUT.tmp" "$OUT"
  done
done
ls -l "$ROOT"/models/*.gguf
