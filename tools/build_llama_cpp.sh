#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$(tr -d '[:space:]' < "$ROOT/tools/llama_cpp_version.txt")"
SRC="$ROOT/third_party/llama.cpp"
mkdir -p "$ROOT/third_party"
if [ -d "$SRC/.git" ] && [ "$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || true)" = "$TAG" ]; then
  echo "llama.cpp $TAG present"
else
  rm -rf "$SRC"
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
if [ "${1:-}" = "--generic" ]; then
  # Same pinned source, generic (non-SIMD) ggml CPU path; used for quantized goldens.
  cmake -S "$ROOT/tools/llama_ref" -B "$ROOT/third_party/build-generic" -DCMAKE_BUILD_TYPE=Release -DNIBBLE_GENERIC=ON
  cmake --build "$ROOT/third_party/build-generic" -j"$(nproc)" --target llama_ref
  exit 0
fi
cmake -S "$ROOT/tools/llama_ref" -B "$ROOT/third_party/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/third_party/build" -j"$(nproc)" --target llama_ref llama-quantize llama-bench
