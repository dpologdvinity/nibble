#pragma once
// Debug-only invariant checks (spec 6.4). NIBBLE_DCHECK(cond, fmt, ...) aborts with
// "<file>:<line>: NIBBLE_DCHECK(<cond>) failed: <message>" when cond is false and
// NDEBUG is not defined. In Release (NDEBUG) it compiles to nothing: cond is only
// named inside sizeof (unevaluated, no code, no allocation) and the message
// arguments are discarded. Use for programmer errors, never for untrusted input,
// which must throw nibble::Error instead.
#ifndef NDEBUG
#include <cstdio>
#include <cstdlib>

#define NIBBLE_DCHECK(cond, ...)                                                 \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: NIBBLE_DCHECK(%s) failed: ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                         \
      std::fputc('\n', stderr);                                                  \
      std::abort();                                                              \
    }                                                                            \
  } while (0)
#else
#define NIBBLE_DCHECK(cond, ...) \
  do {                           \
    (void)sizeof(!(cond));       \
  } while (0)
#endif
