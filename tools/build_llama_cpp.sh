#!/usr/bin/env bash
# Builds the pinned llama.cpp reference tools (the version in tools/llama_cpp_version.txt), cloning the source
# into third_party/ if needed.
#
# Usage: tools/build_llama_cpp.sh [--generic]
#   (default)   third_party/build:         native build (AVX2 etc.) of llama_ref, llama-quantize, llama-bench - used for F16 goldens, making quantized models, and benchmark comparisons
#   --generic   third_party/build-generic: ggml's generic scalar path (no SIMD, no repack) of llama_ref - used for the quantized goldens, because that is the algorithm nibble's scalar code reproduces
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
