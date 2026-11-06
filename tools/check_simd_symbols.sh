#!/usr/bin/env bash
# usage: tools/check_simd_symbols.sh <tier> <object files...>
# Fails if a SIMD tier object file defines a global or weak symbol outside nibble::simd::<tier>::. Such a symbol
# (an inline function, a template instantiation, an Error vtable) would be compiled with the tier's instruction
# set, and the linker may keep that copy for the whole program, so a CPU without the tier would crash in code
# that never meant to use it. Allowed: DW.ref.__gxx_personality_v0 (an exception-handling data reference) and
# AddressSanitizer's __odr_asan.* indicator bytes, which are data, never code.
set -euo pipefail
if (( $# < 2 )); then sed -n 2p "$0" >&2; exit 2; fi
tier=$1; shift
bad=0
for obj in "$@"; do
  # Defined symbols with an upper-case type letter are global (T, D, B, R) or weak (W, V, u); lower case is local.
  while IFS= read -r line; do
    echo "$(basename "$obj"): $line"; bad=1
  done < <(nm -C --defined-only "$obj" | awk -v ns="nibble::simd::${tier}::" '
    $2 ~ /^[A-Zu]$/ {
      name = $0; sub(/^[^ ]+ [^ ]+ /, "", name)
      if (index(name, ns) != 1 && name != "DW.ref.__gxx_personality_v0" && index(name, "__odr_asan.") != 1)
        print $2 " " name
    }')
done
if (( bad )); then echo "check_simd_symbols: symbols above escape nibble::simd::${tier}::" >&2; exit 1; fi
echo "check_simd_symbols: $# objects of tier ${tier} define symbols only in nibble::simd::${tier}::"
