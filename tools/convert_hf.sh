#!/usr/bin/env bash
# Usage: tools/convert_hf.sh <llama32-1b|tinyllama-1.1b>
# Downloads the HF repo (json, safetensors, tokenizer.model) and converts it to an F16 GGUF with
# the pinned llama.cpp converter in a CPU-torch venv (third_party/py). Skips steps whose output exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: convert_hf.sh <llama32-1b|tinyllama-1.1b>}"
case "$NAME" in
  llama32-1b)     REPO=unsloth/Llama-3.2-1B-Instruct;         OUT="$ROOT/models/llama-3.2-1b-instruct-f16.gguf" ;;
  tinyllama-1.1b) REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0;    OUT="$ROOT/models/tinyllama-1.1b-chat-f16.gguf" ;;
  *) echo "error: unknown model '$NAME' (known: llama32-1b tinyllama-1.1b)" >&2; exit 1 ;;
esac
if [ -f "$OUT" ]; then echo "skip $OUT"; exit 0; fi

PY="$ROOT/third_party/py"
if [ ! -x "$PY/bin/python" ]; then
  rm -rf "$PY"; python3 -m venv "$PY"
fi
if [ ! -f "$PY/.deps-ok" ]; then
  "$PY/bin/python" -m pip install -r "$ROOT/third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
  touch "$PY/.deps-ok"
fi

HF="$ROOT/models/hf/$NAME"
mkdir -p "$HF"
FILES=$(curl -L --fail -s "https://huggingface.co/api/models/$REPO" | "$PY/bin/python" -I -c '
import json, sys
for s in json.load(sys.stdin)["siblings"]:
    f = s["rfilename"]
    if "/" not in f and (f.endswith(".json") or f.endswith(".safetensors") or f == "tokenizer.model"):
        print(f)')
[ -n "$FILES" ] || { echo "error: no files listed for $REPO" >&2; exit 1; }
for f in $FILES; do
  curl -L --fail -C - -o "$HF/$f" "https://huggingface.co/$REPO/resolve/main/$f"
done

"$PY/bin/python" "$ROOT/third_party/llama.cpp/convert_hf_to_gguf.py" "$HF" --outtype f16 --outfile "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
ls -l "$OUT"
