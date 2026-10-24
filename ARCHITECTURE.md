# Architecture

nibble is a from-scratch inference engine for decoder-only transformer language models, written in C++20. It reads
GGUF model files directly, tokenizes text itself, runs the forward pass on the CPU with hand-written AVX2 kernels, and
samples tokens. Its goal is to reproduce llama.cpp's results (same tokens, same logits to within rounding) while being
small enough to read end to end.

This document explains how the code is organised, how data flows through it, and why it is built the way it is. For
how to build and use it see [README.md](README.md); for how to work on it see [CONTRIBUTING.md](CONTRIBUTING.md).

## Technology stack

| concern | choice |
|---|---|
| language | C++20; run-time errors are `nibble::Error` exceptions |
| build | CMake ≥ 3.25, one static library, one CLI executable, Catch2 tests |
| runtime dependencies | the C++ standard library and pthreads, nothing else |
| compute | scalar C++ reference kernels plus AVX2 + FMA + F16C kernels (AVX-VNNI where present), chosen at compile time |
| model format | GGUF v2/v3, memory-mapped; F32, F16 and the quantized types Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K |
| architectures | `qwen2` (Qwen2.5) and `llama` (Llama 3.x, TinyLlama) |
| test oracle | a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) release, used only by tests and tools, never linked into nibble |

## Repository layout

```
include/nibble/      the public API (nibble.h, error.h): the only headers an embedding application sees
src/                 the engine
  nibble.cpp           Model / Session implementation: wires everything together, runs the generate loop
  chat.{h,cpp}         chat templates (ChatML, Llama 3, Zephyr) and the Chat conversation state machine
  sampler.{h,cpp}      temperature / top-k / top-p sampling
  thread_pool.{h,cpp}  fork-join pool, default thread-count policy
  quant.{h,cpp}        quantized block formats and the scalar reference kernels (the oracle)
  tensor.{h,cpp}, fp16.h   tensor view, dtype table, half-precision conversion
  gguf/                GGUF parser and mmap wrapper
  tokenizer/           byte-level BPE and SentencePiece tokenizers, pre-tokenizer, Unicode helpers
  model/               LlamaConfig, Weights (tensor binding), Transformer (the forward pass)
  backend/             Backend interface and the CPU implementation
    cpu/                 CpuBackend, ref_ops (scalar kernels)
    cpu/simd/            AVX2 kernels: float matmul, quantized dots, tiled GEMM, elementwise ops
  bench/               memory-bandwidth probe used by `nibble bench --mem`
  util/                escape_control (safe printing), FunctionRef
cli/                 the `nibble` command: info, run, chat, bench
tests/               Catch2 tests, fixtures, golden data, fuzz targets
tools/               scripts and programs around the llama.cpp oracle (model/golden generation, benchmarks)
bench/               benchmark results (JSONL) and the plotting script
docs/bench/          generated charts
```

## Layers and component boundaries

```
 application / CLI
        │  nibble.h:  Model::load, Session::generate, Chat::send
        ▼
 ┌───────────────────────────── src/nibble.cpp ─────────────────────────────┐
 │  Model::Impl            Session::Impl                     Chat (chat.cpp) │
 │  owns file, config,     owns Transformer (KV cache),      template → parts │
 │  tokenizer, weights,    drives prefill / decode loop,     → Session        │
 │  thread pool, backend   runs the Sampler, streams UTF-8                    │
 └──────┬───────────┬───────────────┬─────────────────────────────────────────┘
        │           │               │
   gguf::File   tok::Tokenizer   model::Transformer ──► Backend (abstract)
   (mmap, zero  (BPE / SPM)      (layer loop,                │
    copy)                         KV cache)                  ▼
                                                        CpuBackend ──► ThreadPool
                                                          │      │
                                                    ref_ops.cpp  simd/*.cpp
                                                  (scalar oracle) (AVX2 kernels)
```

The boundaries that matter:

* **Public API vs engine.** Applications include only `nibble/nibble.h`. `Model`, `Session` and `Chat` hide their state
  behind `Impl` structs, so nothing in `src/` leaks. `run` and `chat` in the CLI use only this API; `bench` is the one
  deliberate exception, because it must time the transformer without tokenizer and sampler costs.
* **Model code vs backend.** `model/transformer.cpp` is written only against the abstract `Backend` interface
  (`backend/backend.h`): matmul, embed, rmsnorm, rope, attention, silu_mul, add, bias, copy. It never touches
  activation memory directly (a GPU backend would hand out device pointers). `CpuBackend` is the only implementation
  today; the planned CUDA backend slots in behind the same interface.
* **Backend vs kernels.** `CpuBackend` owns scratch memory and decides *which* kernel runs; the kernels in
  `ref_ops.cpp` and `simd/` only compute. Every SIMD kernel has a scalar twin in `ref_ops.cpp`/`quant.cpp`, and tests
  require the two to agree (bit for bit where promised).
* **File format vs everything else.** Only `gguf/` knows the container format. It hands out `Tensor` views
  (`tensor.h`) and typed metadata; nothing downstream parses bytes from the file.

## Data flow

### Loading a model (`Model::load`)

1. `gguf::MappedFile` maps the file read-only; `gguf::File::parse` validates the header, metadata and tensor table
   against the file length. Nothing is copied: tensor data pointers and array views point into the mapping.
2. `LlamaConfig::from_gguf` reads and checks hyperparameters (layers, widths, heads, rope, epsilon).
3. The vocabulary size is compared with the embedding matrix before the tokenizer builds its tables.
4. `tok::Tokenizer::from_gguf` builds the vocabulary maps, merge ranks, special-token list and end-of-generation set.
5. A `ThreadPool` and `CpuBackend` are created; `Weights::bind` resolves every tensor by name and checks shape and
   type against what the backend supports.

Everything in `Model::Impl` is declared in dependency order, so destruction can never leave a view dangling.

### Generating text (`Session::generate`)

1. The prompt parts are tokenized (special-token text is only parsed in parts marked `special`; BOS is added when the
   model wants it).
2. **Prefill:** the new tokens go through `Transformer::forward` in chunks of up to 512. For each layer:
   RMSNorm → one fused Q/K/V projection (K and V are written straight into the KV cache rows) → bias → rotary
   embedding → attention over the cache → output projection → residual add → RMSNorm → fused gate/up projection →
   SwiGLU → down projection → residual add. Only the last token's hidden state is projected to logits.
3. **Decode loop:** the `Sampler` picks a token from the logits; an end-of-generation token stops the loop; otherwise
   the token's bytes go through a `Utf8Streamer` (so a multi-byte character split across tokens is emitted whole) to the
   caller's callback, and the token is forwarded through the transformer to produce the next logits.
4. `Session` tracks `n_past` (tokens in the KV cache). While a forward is in flight it is set to 0, so an exception
   inside the model leaves a consistent "empty cache" state rather than a half-updated one.

`Chat` sits on top: it formats each turn with the model's template (system/user text is never parsed for control
tokens, so users cannot inject markup), appends only the new turn's tokens to the existing cache, and detects misuse of
the underlying `Session` ("chat context lost").

## Matmul and the quantized-activation scheme

All heavy compute is matrix-vector (decode) or matrix-matrix (prefill) products of a weight matrix with activations.
For quantized weights, nibble follows ggml: it does not dequantize the weights. Instead it quantizes the *activation*
rows to a companion 8-bit format and computes block-wise integer dot products, scaling each block's integer sum by the
two blocks' scales.

| weight type | activation type | block (elements / bytes) |
|---|---|---|
| Q4_0, Q5_0, Q8_0 | Q8_0 | 32 / 18, 22, 34 |
| Q5_1 | Q8_1 (Q8_0 plus the block sum) | 32 / 24 |
| Q4_K, Q5_K, Q6_K | Q8_K (float scale, per-16 sums) | 256 / 144, 176, 210 |

`CpuBackend::matmul_multi` takes a group of weights that share one input (Q/K/V, or gate/up), quantizes the input
once per distinct activation type needed, and runs all weights' rows in a single `parallel_for`. This is also how
mixed-type groups work: real k-quant files use different types for, say, `attn_v` and `attn_q`.

For prefill (`n ≥ 4` tokens) Q4_0/Q8_0 weights use a tiled GEMM that processes 4 weight rows × 16 tokens per tile
over a packed, interleaved copy of the activations (`gemm.cpp`); float weights use a tiled float GEMM. The other
types fall back to the row kernels.

## Numerical fidelity: the oracle design

The central engineering decision is *bit-exactness by construction*, in three links:

1. **ggml generic → scalar code.** `quant.cpp` reproduces llama.cpp's generic (non-SIMD) quantizers and dot products,
   *including the order of floating-point operations*. Float addition is not associative, so the same arithmetic in
   a different order gives different bits. Golden vectors (`tests/data/quant_golden.inc`, generated by
   `tools/quant_vectors` calling the pinned ggml) pin dequantization, quantization and dots bit for bit.
2. **Scalar → AVX2.** The integer part of each block dot is exact in both implementations, so only the float
   evaluation order matters, and the SIMD kernels are written to follow the scalar's order. The k-quant dots need a
   special byte permutation so that each int32 lane holds exactly ggml's `aux32[l]` elements (documented in
   `kquant_avx2.cpp`). The translation units involved are compiled with `-ffp-contract=off` so the compiler cannot
   fuse multiplies and adds. Tests compare results by bit pattern, on unaligned data, with awkward sizes.
3. **Row kernel → GEMM → threads.** Each output element goes through the same operation sequence no matter how the
   work is tiled or split across threads, so results do not depend on the thread count, batch size or chunking.

Non-finite data is handled deliberately: blocks from a file are untrusted, so NaN/Inf scales and all-ones bytes must be
safe to process (no undefined behaviour; the result is non-finite). A NaN in an activation block gives the block a NaN
scale (and zero quants), so every dot that reads it is NaN. Quantizers saturate before the int8 cast, and the AVX2 sign
trick relies on activations never containing -128, which the quantizers guarantee.

Elementwise float kernels (RMSNorm, SiLU, softmax/attention, float matmul) are *not* bit-identical to the scalar code
(different summation order, a polynomial `exp`); tests bound their error instead (≈1e-6 relative).

## Memory and threading model

* **Zero-copy weights.** Weights live in the page cache through `mmap`; loading is O(metadata), not O(model).
* **No allocation in the hot path.** `Transformer` allocates the KV cache and all activation buffers once;
  `Backend::reserve` sizes the attention scores and quantized-activation scratch up front. After a warm-up, decoding
  allocates nothing; `alloc_tests` proves it with a counting `operator new`.
* **Thread pool.** `ThreadPool::parallel_for` is a fork-join: the caller is worker 0, chunks are claimed from an atomic
  cursor, workers spin ~50 µs on a generation counter and then sleep on a futex. Exceptions in a chunk are captured
  and rethrown on the caller after all threads finish. The default thread count reads
  `/sys/devices/system/cpu/*/cpu_capacity` to detect hybrid CPUs and falls back to 2/3 of the hardware threads.
* **One Session per Model at a time** (the backend's attention scratch is shared); a `Session` must not outlive its
  `Model`.

## Error handling and untrusted input

* GGUF files are untrusted. The parser bounds every count and size by the actual file length with overflow-safe
  arithmetic, never reserves more than a few thousand entries from a claimed count, and rejects anything inconsistent
  with an error naming the key or tensor. Text taken from a file is passed through `escape_control` before it reaches an
  error message or the terminal. A libFuzzer target exercises parser and loader.
* Two kinds of failure: run-time problems caused by input (bad file, bad parameters, empty prompt, out of memory while
  loading) throw `nibble::Error`; violated internal invariants abort through `NIBBLE_DCHECK` in debug builds and compile
  to nothing in release.

## Testing strategy

| layer | what it proves | where |
|---|---|---|
| unit / property | each module in isolation, including hostile inputs | `tests/test_*.cpp` |
| golden vectors | quantization, dequantization and dots equal ggml's, bit for bit | `test_quant.cpp`, `quant_golden.inc` |
| kernel equivalence | AVX2 == scalar (bitwise or within tolerance) | `test_simd.cpp`, `test_gemm.cpp`, `test_ops.cpp` |
| model | tiny synthetic models vs an independent naive double-precision forward pass | `test_model.cpp`, `naive_llama.h` |
| allocation | decode allocates nothing; parser memory is bounded; OOM becomes `Error` | `alloc_tests.cpp` |
| end to end | logits of real models vs llama.cpp at every position of a 64-token generation, three prompts per model | `test_e2e_golden.cpp` |
| fuzzing | the parser and loader never crash | `tests/fuzz/` |

End-to-end gates (cosine similarity of logits per position, top-1 agreement) depend on the model type: F16 models must
match to 0.999 with identical argmax except at near-ties; quantized models use 0.995 because activation requantization
amplifies tiny rounding differences (independent correct implementations agree to only about 0.997 cosine at the worst positions). Llama-3.2 k-quant files use
a per-position floor of 0.97 because llama.cpp's own generic and native builds differ by more than 0.995 on them, and the
other k-quant files may have one position between 0.99 and 0.995. The quantized goldens come from llama.cpp's *generic*
build with an F32 KV cache and flash attention off, because that is the algorithm nibble reproduces. End-to-end tests need
multi-gigabyte model files and are not run in CI; everything else is.

## Key decisions and trade-offs

* **Compile-time ISA selection.** A binary built with `-march=native` runs only on CPUs at least as capable as the
  build host. This keeps the dispatch cost zero and the code simple; runtime dispatch is future work.
* **Scalar twins instead of intrinsics-only.** Every optimized kernel keeps a readable twin. This costs code but makes
  correctness checkable and keeps non-AVX2 targets working.
* **The k-quant GEMM tile is not implemented.** Prefill for Q5_0 and the k-quants uses the row kernels. Whether a tiled
  GEMM is worth it is decided by measurement (see README performance notes).
* **No dequantize-to-float path for matmul.** It would be simpler but slower and would not match llama.cpp's numerics.
* **Pinned oracle.** The llama.cpp release in `tools/llama_cpp_version.txt` is the single reference; bumping it means
  regenerating goldens and re-reviewing numerical differences.

## Where to go next

Planned work is a CUDA backend behind the existing `Backend` interface, runtime CPU dispatch, and tiled GEMM kernels for
the k-quants, subject to measurements.
