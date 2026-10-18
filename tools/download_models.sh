#!/usr/bin/env bash
# Usage: tools/download_models.sh [name...]   (no args = all known models)
# Llama 3.2 / TinyLlama are converted from HF safetensors instead: tools/convert_hf.sh <llama32-1b|tinyllama-1.1b>
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models"

declare -A URLS=(
  [qwen2.5-0.5b-f16]="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
)

names=("$@")
if [ ${#names[@]} -eq 0 ]; then names=("${!URLS[@]}"); fi

mkdir -p "$dest"
for name in "${names[@]}"; do
  url="${URLS[$name]:-}"
  if [ -z "$url" ]; then
    echo "error: unknown model '$name' (known: ${!URLS[*]})" >&2
    exit 1
  fi
  file="$dest/$(basename "$url")"
  echo "downloading $name -> $file"
  curl -L --fail -C - -o "$file" "$url"
  if [ ! -s "$file" ]; then
    echo "error: downloaded file is empty: $file" >&2
    exit 1
  fi
done
