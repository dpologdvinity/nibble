#!/usr/bin/env bash
# A stand-in C++ compiler for tests/vnni_probe_check.cmake: behaves like $REAL_CXX (default g++), except that it
# fails on any source file using the AVX-VNNI intrinsic _mm256_dpbusd_avx_epi32. That is what a toolchain whose
# compiler accepts -mavxvnni but whose assembler cannot encode vpdpbusd (GCC 11/12 with binutils < 2.36) does.
set -euo pipefail
for arg in "$@"; do
  if [[ -f $arg ]] && grep -q '_mm256_dpbusd_avx_epi32' "$arg" 2>/dev/null; then
    echo "fake_cxx_no_vnni_intrinsic: cannot encode vpdpbusd" >&2
    exit 1
  fi
done
exec "${REAL_CXX:-g++}" "$@"
