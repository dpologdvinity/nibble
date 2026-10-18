# nibble

nibble is a from-scratch LLM inference engine in C++20. It loads GGUF model
files directly (mmap, no conversion step), implements its own byte-level BPE
tokenizer and transformer forward pass, and is checked against llama.cpp as a
reference. It has no runtime dependencies beyond the C++ standard library and
pthreads.

## Status

Milestones M0–M4 are done: Qwen2.5 models in F16, Q8_0 and Q4_0 run on the CPU
with scalar reference kernels, with a sampler and a multi-turn `nibble chat`.
This is a scalar baseline; optimization comes in M6.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Requires CMake ≥ 3.25 and g++ 13 or clang 18. Catch2 is fetched at configure time.

## Getting a model

```sh
tools/download_models.sh   # downloads qwen2.5-0.5b-instruct-fp16.gguf into models/
tools/make_quants.sh       # makes the Q8_0 and Q4_0 models from it (needs the llama.cpp build)
```

## Usage

```sh
build/nibble info models/qwen2.5-0.5b-instruct-fp16.gguf
build/nibble run -m models/qwen2.5-0.5b-instruct-fp16.gguf -p "The capital of France is" -n 32
build/nibble chat -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

`nibble info <model.gguf>` prints the GGUF metadata and tensor table.

`nibble run` flags:

| flag | meaning |
|------|---------|
| `-m <model.gguf>` | model file (required) |
| `-p <prompt>` | prompt text (exactly one of `-p` / `-f`) |
| `-f <file>` | read the prompt from a file |
| `-n <N>` | maximum tokens to generate (default 256) |
| `-t <N>` | worker threads (default: hardware concurrency) |
| `-c <N>` | context length (default: min(4096, model context)) |
| `--temp <f>` | sampling temperature, >= 0; 0 is greedy (default 0.8) |
| `--top-k <N>` | top-k cutoff; 0 disables (default 40) |
| `--top-p <f>` | nucleus cutoff in (0, 1] (default 0.95) |
| `--seed <u64>` | sampler seed (default: random) |

Text streams to stdout; a stop reason and timing line go to stderr. Bad flag
values print usage and exit 2; errors such as a missing model exit 1.

`nibble chat -m <model.gguf> [--system <text>] [-n N] [-t N] [-c N]` plus the
sampling flags reads one user message per stdin line and streams each reply to
stdout (`-n` is per reply, default 512). Blank lines are skipped. Qwen2 models
only (ChatML). It prompts with `> ` on stderr only when stdin is a terminal,
stops at EOF, and on a full context prints `[context full]` and exits 0.

## Tests

```sh
ctest --test-dir build -LE e2e
```

End-to-end tests compare against llama.cpp. They need the model, a pinned
llama.cpp build, and golden files:

```sh
tools/build_llama_cpp.sh && tools/build_llama_cpp.sh --generic && tools/make_quants.sh \
  && for m in f16 q8_0 q4_0; do tools/make_golden.sh qwen2.5-0.5b-$m; done \
  && ctest --test-dir build -L e2e
```

## Correctness

Against llama.cpp on Qwen2.5-0.5B-Instruct F16:

- Tokenizer: 37/37 corpus cases tokenize exactly as llama.cpp does.
- Logits: minimum cosine similarity 0.99996 over all positions.
- Greedy decoding: 64/64, 64/64 and 46/64 tokens match on the three e2e prompts.
  The third diverges at a near-tie, where an fp64 reference implementation agrees
  with nibble.

Q8_0 and Q4_0 (made with `llama-quantize --pure`) are compared against a
generic-path llama.cpp build (no SIMD, llamafile or weight repack) with an F32 KV
cache. The gate is looser than for F16: cosine >= 0.995 at every position and on
the prefill last row, and top-1 agreement on >= 90% of positions per prompt
(near-ties count as agreement). Greedy decoding is reported but not asserted.
Activation requantization amplifies rounding-level differences: nibble, generic
llama.cpp and an fp64 oracle with Q8_0 requantization disagree pairwise at about
0.9966 cosine, so a tighter gate is not attainable. Minimum cosine (capital / code / chat):

- Q8_0: 0.99950 / 0.99875 / 0.99672
- Q4_0: 0.99888 / 0.99858 / 0.99658

## Roadmap

M4: Q8_0/Q4_0 quantized weights, sampler, chat template · M5: Llama 3.2 and
TinyLlama · M6: SIMD kernels, threading and benchmarks · Phase 2: CUDA backend.

## License

MIT; see [LICENSE](LICENSE).
