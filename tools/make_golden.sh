#!/usr/bin/env bash
# Usage: tools/make_golden.sh <model-name>   (model: models/<model-name>.gguf-resolved below)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: make_golden.sh <model-name>}"
case "$NAME" in
  qwen2.5-0.5b-f16) GGUF="$ROOT/models/qwen2.5-0.5b-instruct-fp16.gguf" ;;
  *) GGUF="$ROOT/models/$NAME.gguf" ;;
esac
[ -f "$GGUF" ] || { echo "missing model $GGUF" >&2; exit 1; }
REF="$ROOT/third_party/build/bin/llama_ref"
OUT="$ROOT/models/golden/$NAME"
mkdir -p "$OUT"
"$REF" tokenize "$GGUF" "$ROOT/tests/tokenizer/corpus.txt" "$OUT/tokenize.txt"
for p in capital code chat; do
  "$REF" logits "$GGUF" "$ROOT/tests/e2e/prompts/$p.txt" 64 "$OUT/$p.bin"
done
ls -l "$OUT"
