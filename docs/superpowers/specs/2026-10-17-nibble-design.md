# nibble — Design Spec

**Date:** 2026-10-17
**Status:** Approved in brainstorming; pending written-spec review

## 1. Purpose

nibble is a from-scratch LLM inference engine in C++. It serves two goals:

1. **Learning / portfolio:** implement every layer (file format, tokenizer, transformer forward pass, quantized SIMD kernels, threading) and document measured performance gains step by step.
2. **Usable tool:** code clean enough to grow into something people embed or run — a library with a stable public API plus a CLI.

### Success criteria (v1)

- Runs Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, and TinyLlama-1.1B from GGUF files on CPU.
- Output matches llama.cpp on the same GGUF and prompt (criteria in §8.3).
- Decode throughput (`tg128`) within ~1.5× of llama.cpp CPU on Q4_0 on the development machine; matching llama.cpp is a stretch goal.
- Every optimization step is benchmarked and charted in the README, including a memory-bandwidth roofline.

## 2. Scope

### In scope (v1)

- **Architectures:** Llama family — `general.architecture` ∈ {`llama`, `qwen2`}.
- **File format:** GGUF v2 and v3 only.
- **Tensor types:** F32, F16, Q8_0, Q4_0.
- **Hardware:** x86-64 CPU with AVX2/FMA/F16C; AVX-VNNI path when available at compile time; scalar fallback always built.
- **Interfaces:** C++ library `libnibble` (public header `include/nibble/nibble.h`) and CLI `nibble` (`run`, `bench`, `info`).
- **Platform:** Linux (including WSL2), g++ ≥ 13, C++20, CMake.

### Out of scope (v1)

HTTP/OpenAI-compatible server, multi-request batching, Python bindings, Windows/macOS, ARM NEON, runtime CPU-feature dispatch, Jinja chat-template engine, sliding-window context, K-quants.

### Planned follow-ups

- **v1.1:** Q4_K and Q6_K (loads the common Q4_K_M files).
- **Phase 2:** CUDA backend, developed and benchmarked on Google Colab / Kaggle GPUs (T4, P100). The user runs notebooks; the repo provides a build-and-bench notebook cell.

### Development machine

Intel i7-1265U (2 P-cores + 8 E-cores, 12 threads), AVX2 + FMA + F16C + AVX-VNNI, no AVX-512, 15 GB RAM.

## 3. Architecture

Model code is written against a small op layer behind a `Backend` interface. There is no compute graph or scheduler. `CpuBackend` ships in v1; `CudaBackend` is the planned second implementation in Phase 2.

### 3.1 Layout

```
nibble/
  include/nibble/nibble.h     public API (only header users see)
  src/
    gguf/        GGUF reader: mmap file, parse header + metadata + tensor table
    tensor.h     Tensor = {dtype, shape[<=4], data ptr}; borrowed (mmap) or owned
    tokenizer/   BPE from GGUF vocab: byte-level (Qwen/Llama 3), SPM-style (TinyLlama)
    model/       LlamaConfig from metadata; weight binding by GGUF name; forward()
    backend/     Backend interface; cpu/ = scalar reference + AVX2/VNNI kernels
    kv_cache     per-layer K/V buffers [n_ctx, n_kv_heads, head_dim]
    sampler      greedy, temperature, top-k, top-p, seeded RNG
    thread_pool  persistent workers, dynamic chunking
  cli/           nibble run | bench | info
  tests/         unit, tokenizer, e2e (golden), fuzz
  bench/         benchmark harness + results/
  tools/         llama.cpp reference build + golden-dump + model download scripts
  models/        downloaded GGUFs (gitignored)
```

### 3.2 Unit boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `gguf` | Parse and validate GGUF; expose metadata and tensor views | OS mmap |
| `tokenizer` | Encode text to token IDs, decode IDs to UTF-8 | `gguf` metadata only |
| `model` | Llama architecture: config, weight binding, forward pass expressed as ops | `Backend`, `tensor`, `kv_cache` |
| `backend/cpu` | All math and SIMD; sees tensors, never models | `tensor`, `thread_pool` |
| `sampler` | Logits to next token | none |
| public API | `Model`, `Session`, `generate` | everything above; hides internals |

### 3.3 Backend ops

`embed`, `rmsnorm`, `matmul` (dispatches on weight dtype), `add` (incl. bias), `rope`, `attention` (causal, GQA), `silu_mul`, `softmax`. Fused ops (e.g. rmsnorm + activation quantize) are added as explicit named ops when the optimization ladder needs them.

All ops accept activations shaped `[n_tokens, dim]` so prefill uses GEMM from the start.

### 3.4 Public API (shape, not final signatures)

```cpp
namespace nibble {
struct ModelParams   { int n_ctx = 0; int n_threads = 0; };   // 0 = auto
struct GenParams     { int max_tokens; float temperature; int top_k; float top_p;
                       uint64_t seed; bool chat; };
enum class StopReason { Eos, MaxTokens, ContextFull, Cancelled };

class Model   { public: static Model load(const std::string& path, ModelParams); };
class Session { public:
  explicit Session(Model&);
  // on_token receives complete UTF-8; return false to cancel.
  StopReason generate(std::string_view prompt, const GenParams&,
                      const std::function<bool(std::string_view)>& on_token);
};
class Error : public std::runtime_error { using runtime_error::runtime_error; };
}
```

### 3.5 Architecture-specific handling

- **Qwen2:** Q/K/V projection biases (`blk.N.attn_{q,k,v}.bias`); RoPE NEOX mode (rotate halves).
- **Llama:** no biases; RoPE normal mode (adjacent pairs — GGUF converter already permuted weights).
- **Llama 3.2:** optional `rope_freqs.weight` frequency-scaling tensor; applied when present.
- **Tied embeddings:** if `output.weight` is absent, use `token_embd.weight`.
- **Chat templates:** hardcoded ChatML (Qwen) and Llama 3 format, chosen from GGUF metadata. TinyLlama uses its Zephyr-style format. Raw-prompt mode always available.

## 4. Data flow

### 4.1 Load — `Model::load(path, params)`

1. mmap the file read-only. Validate magic `GGUF` and version ∈ {2, 3}. Parse metadata KV pairs and the tensor table; apply `general.alignment` (default 32).
2. Require `general.architecture` ∈ {`llama`, `qwen2`}. Build `LlamaConfig` from `<arch>.block_count`, `.embedding_length`, `.feed_forward_length`, `.attention.head_count`, `.attention.head_count_kv`, `.rope.freq_base`, `.attention.layer_norm_rms_epsilon`, `.context_length`.
3. Bind each weight by name (`token_embd.weight`, `blk.{i}.attn_q.weight`, …, `output_norm.weight`, `output.weight`). Verify shape and dtype against the config. Weights are zero-copy views into the mapping.
4. Allocate the KV cache (`n_ctx` = `params.n_ctx` or `min(4096, context_length)`) and a scratch arena sized for the max prefill batch. No allocation occurs in the generation loop.

### 4.2 Generate — `Session::generate(prompt, params, on_token)`

1. Apply chat template if `params.chat`; tokenize.
2. **Prefill:** run all prompt tokens as one batch; compute logits for the last position only.
3. **Decode:** one token per step → sample → detokenize → `on_token`. Stop on EOS, `max_tokens`, full context, or callback returning `false`.

### 4.3 One transformer layer (activations F32)

```
h      = rmsnorm(x, attn_norm)
q,k,v  = matmul(h, Wq|Wk|Wv) (+ bias if qwen2)
q,k    = rope(q, k, pos)                // mode per architecture
cache[layer][pos] = k, v
a      = attention(q, cache[0..pos])    // causal; n_heads / n_kv_heads query heads share each K/V head
x     += matmul(a, Wo)
h      = rmsnorm(x, ffn_norm)
x     += matmul(silu(matmul(h, Wgate)) * matmul(h, Wup), Wdown)
```

After the last layer: `logits = matmul(rmsnorm(x, output_norm), Woutput)`.

## 5. Kernels and performance

### 5.1 Quantized matmul

- **Q4_0 / Q8_0 weights:** quantize the activation row to Q8_0 blocks, then integer block dot products — AVX2 `_mm256_maddubs_epi16`, AVX-VNNI `vpdpbusd` — scaled by the F16 block scales converted to F32.
  - Q4_0 block (18 bytes): F16 `d`, 16 bytes of nibbles. Low nibbles are elements 0–15, high nibbles 16–31. Value = `(q − 8) · d`.
  - Q8_0 block (34 bytes): F16 `d`, 32 × `int8`. Value = `q · d`.
- **F16 weights:** F16C conversion + FMA. **F32 weights:** FMA.

### 5.2 Kernel rules

1. Every op has a scalar reference implementation; it is the correctness oracle.
2. Every SIMD kernel is tested against its scalar reference.
3. ISA is selected at compile time (`-march=native` default; CMake option to force scalar).

### 5.3 Threading

Persistent worker pool. Matmuls are split by output rows into chunks claimed through an atomic counter, so P-cores take more chunks and E-cores do not create a tail. Attention is split per head. Thread count is set by `-t`. Until M6 the default is `std::thread::hardware_concurrency()`; M6 replaces it with the best value from a `nibble bench` thread sweep on the dev machine.

### 5.4 Performance model

Decode is memory-bandwidth-bound: each token reads all weights. Qwen2.5-0.5B Q4_0 is ~350 MB, so at ~40 GB/s the ceiling is ~110 tok/s. `nibble bench` measures peak bandwidth with a streaming-read test and reports achieved bandwidth as a percentage of it.

Benchmarks use llama-bench naming: `pp512` (prefill 512 tokens) and `tg128` (generate 128 tokens).

### 5.5 Optimization ladder

Each step is benchmarked against the previous one and charted in the README:

1. Scalar reference
2. Thread pool
3. AVX2 dequantize + FMA
4. Q8_0 activations + integer dot
5. AVX-VNNI
6. Prefill GEMM tiling (cache-blocked microkernel)
7. Fused ops (rmsnorm + quantize) and attention tuning

KV cache is F32 in v1; F16 KV is a later optimization.

## 6. Error handling

### 6.1 Untrusted input

GGUF files are untrusted. The parser:

- bounds-checks every read against the file size;
- uses overflow-checked arithmetic for sizes, offsets, and dimension products;
- caps string lengths and array counts by remaining bytes;
- requires tensor data to lie within the file and respect alignment;
- rejects unknown tensor types and unknown metadata value types.

The parser has a libFuzzer target and runs under ASan/UBSan in CI.

### 6.2 Reporting

- **Load/config errors:** throw `nibble::Error` with a specific message, e.g.
  `unsupported tensor type Q4_K for blk.0.attn_q.weight — v1 supports F32/F16/Q8_0/Q4_0; convert with llama-quantize`.
- **Generation:** returns `StopReason`; expected conditions (EOS, max tokens, full context, cancellation) do not throw.
- **Allocation failure:** KV cache allocation failure reports the required bytes and suggests a smaller `n_ctx`.
- **CLI:** catches `nibble::Error`, prints it, exits non-zero.

### 6.3 Streaming UTF-8

The detokenizer buffers incomplete multi-byte sequences so `on_token` always receives complete, valid UTF-8.

### 6.4 Debug checks

Debug builds assert shape agreement at every op call. Release builds rely on load-time shape validation and skip per-op checks.

## 7. Dependencies and tooling

- **Library:** C++20 standard library and `std::thread` only.
- **Tests:** a test framework fetched by CMake (Catch2 v3).
- **Fuzzing:** clang + libFuzzer.
- **Reference:** llama.cpp, built locally by `tools/` for `llama-tokenize`, logit dumps, `llama-quantize`, and `llama-bench`. Not a build dependency of nibble.
- **Models:** downloaded from Hugging Face by a `tools/` script into gitignored `models/`.

## 8. Testing

### 8.1 Unit

- GGUF parser on tiny synthetic files produced by a test-only GGUF writer, including corrupted variants.
- Q4_0/Q8_0 block decode against hand-computed values.
- Each SIMD kernel against its scalar reference on random inputs, with tolerance.
- `rope`, `rmsnorm`, `softmax`, `attention` against naive implementations.

### 8.2 Tokenizer

- Token IDs exactly match `llama-tokenize` on a fixed corpus: Unicode, emoji, whitespace runs, digits, code, CJK.
- `decode(encode(s)) == s` on the same corpus.

### 8.3 End-to-end (golden)

`tools/` dumps llama.cpp reference logits for fixed prompts per model. A model passes when:

- top-1 token agrees at every position;
- logit cosine similarity ≥ 0.999 at every position;
- 64 greedy tokens are identical, except that a divergence is allowed at a position where the reference top-2 logit gap is < 1e-3.

E2E tests need ~2 GB of models and run locally via `ctest -L e2e`; they are not in CI.

### 8.4 Performance

`nibble bench` writes `pp512`/`tg128` results to `bench/results/`. Tracked over time; never fails CI.

### 8.5 CI

Build, unit tests, ASan/UBSan build + tests, short fuzz run.

## 9. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | CMake, Catch2, CI, thread pool | `ctest` green |
| M1 | GGUF reader + `nibble info` | prints Qwen2.5 metadata and tensors; fuzzer clean |
| M2 | Byte-level BPE tokenizer | IDs match `llama-tokenize` on corpus |
| M3 | Scalar forward pass, Qwen2.5-0.5B F16 | golden match; first coherent text |
| M4 | Q8_0/Q4_0, sampler, `nibble run` + chat | Q4_0 golden match; streaming chat works |
| M5 | Llama-3.2-1B (`rope_freqs`) + TinyLlama (SPM tokenizer) | golden match for both |
| M6 | Optimization ladder + bench + README charts | each step measured; `tg128` within ~1.5× of llama.cpp |
| v1.1 | Q4_K / Q6_K | Q4_K_M files load and golden-match |
| P2 | CUDA backend (Colab/Kaggle) | matches CPU logits; T4 benchmarks published |
