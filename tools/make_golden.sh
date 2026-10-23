#!/usr/bin/env bash
# Usage: tools/make_golden.sh [--eog-only | --special-only] <model-name>
#   --eog-only: write only eog.txt (vocab-only load, default build); leaves tokenize/logits goldens alone.
#   --special-only: write only tokenize_special.txt (corpus_special.txt with parse_special, no bos).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EOG_ONLY=0
SPECIAL_ONLY=0
[ "${1:-}" = "--eog-only" ] && { EOG_ONLY=1; shift; }
[ "${1:-}" = "--special-only" ] && { SPECIAL_ONLY=1; shift; }
NAME="${1:?usage: make_golden.sh [--eog-only | --special-only] <model-name>}"
# Golden names map to model files by family: <family>-<x> -> <file base>-<x> (f16 -> fp16 for Qwen).
case "$NAME" in
  qwen2.5-0.5b-f16) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-fp16.gguf" ;;
  qwen2.5-0.5b-*) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-${NAME#qwen2.5-0.5b-}.gguf" ;;
  llama32-1b-*) GGUF="$ROOT/models/llama-3.2-1b-instruct-${NAME#llama32-1b-}.gguf" ;;
  tinyllama-1.1b-*) GGUF="$ROOT/models/tinyllama-1.1b-chat-${NAME#tinyllama-1.1b-}.gguf" ;;
  *) GGUF="$ROOT/models/$NAME.gguf" ;;
esac
[ -f "$GGUF" ] || { echo "missing model $GGUF" >&2; exit 1; }
if [ "$EOG_ONLY" = 1 ]; then
  mkdir -p "$ROOT/models/golden/$NAME"
  "$ROOT/third_party/build/bin/llama_ref" eog "$GGUF" "$ROOT/models/golden/$NAME/eog.txt"
  exit 0
fi
if [ "$SPECIAL_ONLY" = 1 ]; then
  mkdir -p "$ROOT/models/golden/$NAME"
  "$ROOT/third_party/build/bin/llama_ref" tokenize --special "$GGUF" "$ROOT/tests/tokenizer/corpus_special.txt" \
    "$ROOT/models/golden/$NAME/tokenize_special.txt"
  exit 0
fi
# F16 golden: default (native AVX2) build. Quantized goldens: the generic build with an F32 KV cache,
# and flash attention off (--kv-f32 does both: its CPU kernel rounds K/V through F16, which activation
# requantization amplifies), because the native build (AVX2 quantizer/dot, llamafile sgemm, Q4_0 repack, F16 KV) is not the
# generic algorithm nibble implements, and activation requantization amplifies rounding differences
# past the fixed cosine threshold. Build it with: tools/build_llama_cpp.sh --generic
REF="$ROOT/third_party/build/bin/llama_ref"
KV=()
case "$NAME" in
  *-f16) ;;
  *) REF="$ROOT/third_party/build-generic/bin/llama_ref"; KV=(--kv-f32) ;;
esac
[ -x "$REF" ] || { echo "missing $REF" >&2; [ "${KV[*]:-}" ] && echo "run: tools/build_llama_cpp.sh --generic" >&2; exit 1; }
OUT="$ROOT/models/golden/$NAME"
mkdir -p "$OUT"
"$REF" tokenize "$GGUF" "$ROOT/tests/tokenizer/corpus.txt" "$OUT/tokenize.txt"
"$REF" eog "$GGUF" "$OUT/eog.txt"
"$REF" tokenize --special "$GGUF" "$ROOT/tests/tokenizer/corpus_special.txt" "$OUT/tokenize_special.txt"
CHAT=chat   # chat prompt per model family; the output is always chat.bin
case "$NAME" in
  llama32-1b-*) CHAT=chat_llama3 ;;
  tinyllama-*) CHAT=chat_zephyr ;;
esac
for p in capital code chat; do
  src=$p; [ "$p" = chat ] && src=$CHAT
  "$REF" logits "$GGUF" "$ROOT/tests/e2e/prompts/$src.txt" 64 "$OUT/$p.bin" ${KV[@]+"${KV[@]}"}
done
ls -l "$OUT"
