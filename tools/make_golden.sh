#!/usr/bin/env bash
# Usage: tools/make_golden.sh <model-name>   (model: models/<model-name>.gguf-resolved below)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: make_golden.sh <model-name>}"
case "$NAME" in
  qwen2.5-0.5b-f16) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-fp16.gguf" ;;
  qwen2.5-0.5b-q8_0) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-q8_0.gguf" ;;
  qwen2.5-0.5b-q4_0) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-q4_0.gguf" ;;
  *) GGUF="$ROOT/models/$NAME.gguf" ;;
esac
[ -f "$GGUF" ] || { echo "missing model $GGUF" >&2; exit 1; }
# F16 golden: default (native AVX2) build. Quantized goldens: the generic build with an F32 KV cache,
# because the native build (AVX2 quantizer/dot, llamafile sgemm, Q4_0 repack, F16 KV) is not the
# generic algorithm nibble implements, and activation requantization amplifies rounding differences
# past the fixed cosine threshold. Build it with: tools/build_llama_cpp.sh --generic
REF="$ROOT/third_party/build/bin/llama_ref"
KV=()
case "$NAME" in
  qwen2.5-0.5b-q8_0|qwen2.5-0.5b-q4_0) REF="$ROOT/third_party/build-generic/bin/llama_ref"; KV=(--kv-f32) ;;
esac
[ -x "$REF" ] || { echo "missing $REF" >&2; [ "${KV[*]:-}" ] && echo "run: tools/build_llama_cpp.sh --generic" >&2; exit 1; }
OUT="$ROOT/models/golden/$NAME"
mkdir -p "$OUT"
"$REF" tokenize "$GGUF" "$ROOT/tests/tokenizer/corpus.txt" "$OUT/tokenize.txt"
for p in capital code chat; do
  "$REF" logits "$GGUF" "$ROOT/tests/e2e/prompts/$p.txt" 64 "$OUT/$p.bin" ${KV[@]+"${KV[@]}"}
done
ls -l "$OUT"
