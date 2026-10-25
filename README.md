# nibble

[![CI](https://github.com/dpologdvinity/nibble/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/dpologdvinity/nibble/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/dpologdvinity/nibble)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Platform: Linux x86-64](https://img.shields.io/badge/platform-linux--x86--64-lightgrey.svg)](#build)
[![Dependencies: none](https://img.shields.io/badge/runtime%20dependencies-none-brightgreen.svg)](ARCHITECTURE.md)
[![Last commit](https://img.shields.io/github/last-commit/dpologdvinity/nibble/master)](https://github.com/dpologdvinity/nibble/commits/master)
[![Open issues](https://img.shields.io/github/issues/dpologdvinity/nibble)](https://github.com/dpologdvinity/nibble/issues)
[![Code size](https://img.shields.io/github/languages/code-size/dpologdvinity/nibble)](https://github.com/dpologdvinity/nibble)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025e8c.svg?logo=dependabot)](.github/dependabot.yml)

nibble is a from-scratch LLM inference engine in C++20. It loads GGUF model
files directly (mmap, no conversion step), implements its own byte-level BPE
tokenizer and transformer forward pass, and is checked against llama.cpp as a
reference. It has no runtime dependencies beyond the C++ standard library and
pthreads.

nibble targets one job: run quantized open-weight chat models on a CPU, correctly. "Correctly" means it is checked
against llama.cpp at every step: the same token ids from the tokenizer, and logits that agree with llama.cpp's to
within the rounding noise two correct implementations have (see [Correctness](#correctness)). It is a small,
readable code base (about 6,000 lines of library code, comments included) rather than a framework: see
[ARCHITECTURE.md](ARCHITECTURE.md) for how it is put together.

## Status

Milestones M0–M7 are done. Qwen2.5-0.5B, Llama 3.2 1B and TinyLlama 1.1B run on the CPU in F16, Q8_0, Q4_0,
Q5_0, Q5_1 and the k-quants Q4_K, Q5_K and Q6_K (the files llama.cpp's `llama-quantize` makes as `Q4_K_M`,
`Q5_K_M` and `Q6_K`), with AVX2/AVX-VNNI kernels (scalar twins kept as the reference) on a thread pool, a
sampler, and a multi-turn `nibble chat`. See [Performance](#performance) for numbers against llama.cpp.

## Quickstart

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
tools/download_models.sh                                   # Qwen2.5-0.5B-Instruct F16 (about 1.3 GB)
build/nibble chat -m models/qwen2.5-0.5b-instruct-fp16.gguf
```

Type a message and press Enter; the reply streams to the terminal. `nibble --help` lists the other subcommands
(`run`, `info`, `bench`); [Usage](#usage) has the details. To run a smaller, faster quantized file, build it with
`tools/make_quants.sh` (needs the llama.cpp tools; see [Getting a model](#getting-a-model)).

A container image is also available: `docker build -t nibble .`, then
`docker run --rm -it -v "$PWD/models:/models:ro" nibble chat -m /models/<file>.gguf`.

## Supported models

| model | types | how to get it |
|-------|-------|---------------|
| Qwen2.5-0.5B-Instruct | F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M, Q6_K | `tools/download_models.sh` (F16), then `tools/make_quants.sh` (the rest) |
| Llama-3.2-1B-Instruct | F16, Q4_K_M, Q5_K_M, Q6_K | `tools/convert_hf.sh llama32-1b` (F16), then `tools/make_quants.sh` |
| TinyLlama-1.1B-Chat-v1.0 | F16, Q4_K_M, Q5_K_M, Q6_K | `tools/convert_hf.sh tinyllama-1.1b` (F16), then `tools/make_quants.sh` |

**Tensor types the loader accepts:** F32, F16, Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K. The files above are the
ones tested. `Q4_K_M` and `Q5_K_M` are *mixes*: `llama-quantize` stores most matrices as Q4_K / Q5_K but uses Q6_K for
some, Q5_0 / Q5_1 or Q8_0 where a row length is not a multiple of 256 (Qwen2.5-0.5B), and F32 for norms. Other
k-quant mixes should load if they use only the types above, but are not claimed. Files with other types (Q2_K, Q3_K,
Q4_1, IQ*, BF16, ...) are rejected at load with an error naming the type.

`convert_hf.sh` downloads the Hugging Face weights at a pinned commit, checks
every file against the hashes the Hugging Face API lists for that commit, and
converts them offline with the pinned llama.cpp converter. It refuses repos whose
config could make the converter run repo code (`auto_map`, or an unexpected
`tokenizer_class`). `nibble chat` picks the chat format from the model's chat
template: ChatML (Qwen2.5), Llama 3 header format (Llama 3.2) or Zephyr
(TinyLlama). The Llama 3.2 prompt carries the official template's default
knowledge-cutoff and date lines, and trims system and user text as that template
does.

Changes for Qwen since M4: a given `--seed` now gives different output than in
M4, because each reply derives its own seed; generation also stops on `</s>` and
the FIM pad, repo-name and file-separator tokens, as in llama.cpp; and with
special parsing, `</s>` is now a control token.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Requires CMake ≥ 3.25 and g++ 13 or clang 18. Catch2 is fetched at configure time.

The SIMD kernels are chosen at compile time: they need AVX2, FMA and F16C, and the
VNNI dot products also need AVX-VNNI. `NIBBLE_NATIVE` defaults to ON
(`-march=native`), so the binary may not run on an older CPU than the build host.
`-DNIBBLE_NATIVE=OFF` builds a portable binary that runs only the scalar kernels and
is much slower (add `-DCMAKE_CXX_FLAGS=-march=x86-64-v3` for a redistributable AVX2 binary).
`-DNIBBLE_SIMD=OFF` forces the scalar kernels even on a native build. `-DNIBBLE_BUILD_TESTS=OFF`
skips the tests (and the Catch2 download). `nibble info <model.gguf>` prints the active set, e.g.
`kernels: avx2+fma+f16c+avxvnni` or `kernels: scalar`.

## Getting a model

```sh
tools/download_models.sh                      # qwen2.5-0.5b-instruct-fp16.gguf into models/
tools/convert_hf.sh llama32-1b                # Llama-3.2-1B-Instruct F16 (downloads and converts, hash-verified)
tools/convert_hf.sh tinyllama-1.1b            # TinyLlama-1.1B-Chat F16
tools/build_llama_cpp.sh                      # builds the pinned llama.cpp (llama-quantize, llama_ref, llama-bench)
tools/make_quants.sh                          # Q8_0 and Q4_0 of Qwen, and Q4_K_M / Q5_K_M / Q6_K of all three
```

`models/` is git-ignored. Each script skips files that already exist.

## Usage

```sh
build/nibble info models/qwen2.5-0.5b-instruct-fp16.gguf
build/nibble run -m models/qwen2.5-0.5b-instruct-fp16.gguf -p "The capital of France is" -n 32
build/nibble chat -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

`nibble --help` prints the subcommands and `nibble --version` the build's git SHA. `nibble info <model.gguf>`
prints the GGUF metadata and tensor table (a quick way to see which quantization types a file really contains).

`nibble run` flags:

| flag | meaning |
|------|---------|
| `-m <model.gguf>` | model file (required) |
| `-p <prompt>` | prompt text (exactly one of `-p` / `-f`) |
| `-f <file>` | read the prompt from a file |
| `-n <N>` | maximum tokens to generate (default 256) |
| `-t <N>` | worker threads (default: see [Default thread count](#performance)) |
| `-c <N>` | context length (default: min(4096, model context)) |
| `--temp <f>` | sampling temperature, >= 0; 0 is greedy (default 0.8) |
| `--top-k <N>` | top-k cutoff; 0 disables (default 40) |
| `--top-p <f>` | nucleus cutoff in (0, 1] (default 0.95) |
| `--seed <u64>` | sampler seed (default: random); each reply draws its own stream, derived from the seed and the reply's index, so a seeded `chat` is reproducible and its turns are not correlated |

Sampling order is top-k, then temperature-scaled softmax, then top-p. llama.cpp's
default chain is top-k, top-p, min-p (0.05), temperature, so the same flag values
select a different nucleus there.

Text streams to stdout; a stop reason and timing line go to stderr. Bad flag
values print usage and exit 2; errors such as a missing model exit 1.

`nibble chat -m <model.gguf> [--system <text>] [-n N] [-t N] [-c N]` plus the
sampling flags reads one user message per stdin line and streams each reply to
stdout (`-n` is per reply, default 512). Blank lines are skipped. It prompts with
`> ` on stderr only when stdin is a terminal, stops at EOF, and on a full context
prints `[context full]` and exits 0.

`nibble bench -m <model.gguf> [-t 1,2,4] [-p 512] [-n 128] [-r 5] [--mem] [--out f.jsonl]`
measures prefill (`pp<p>`) and decode (`tg<n>`) tok/s per thread count (median, min,
max over `-r` reps after one warm-up); `--mem` adds a memory-bandwidth probe (GB/s means
1e9 bytes per second). `--out` appends one JSON line per measurement (delete the file
to start fresh); each line carries `"kernels"`, the same string `nibble info` prints
(`avx2+fma+f16c+avxvnni`, `avx2+fma+f16c` or `scalar`); rows without the field predate it and are
scalar. `nibble bench --version` prints the baked-in git SHA (`-dirty` if
tracked files differ from HEAD). `tools/bench_llamacpp.sh <model> <threads>
<out.jsonl>` (env `P`, `N`, `R` override prompt tokens=512, gen tokens=128, reps=5; it also appends) produces the same schema from llama.cpp's `llama-bench`. Results live in
`bench/results/`.

## Tests

```sh
ctest --test-dir build -LE e2e    # unit, integration, allocation and CLI tests (about 290 cases, a few minutes)
```

These need no model files and run in CI on every push (see [CONTRIBUTING.md](CONTRIBUTING.md) for the sanitizer,
scalar-kernel and fuzzing variants). The **end-to-end tests** (`-L e2e`) compare against llama.cpp on the real models
and need the model files, a pinned llama.cpp build and golden files, so they run locally:

```sh
tools/build_llama_cpp.sh && tools/build_llama_cpp.sh --generic && tools/make_quants.sh
for m in qwen2.5-0.5b-f16 qwen2.5-0.5b-q8_0 qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m qwen2.5-0.5b-q5_k_m qwen2.5-0.5b-q6_k \
         llama32-1b-f16 llama32-1b-q4_k_m llama32-1b-q5_k_m llama32-1b-q6_k \
         tinyllama-1.1b-f16 tinyllama-1.1b-q4_k_m tinyllama-1.1b-q5_k_m tinyllama-1.1b-q6_k; do
  tools/make_golden.sh $m
done
ctest --test-dir build -L e2e     # 42 model x prompt cases plus tokenizer and file-mix checks
```

Cases skip themselves when a model or golden is missing. The goldens (about 40-50 MB each) are not committed.

## Correctness

Everything below compares nibble with the pinned llama.cpp release (b11715) on the same model file and the same
tokens: at every position of a 64-token greedy generation (three prompts per model) the logits must agree.

**Tokenizer.** 37/37 corpus cases tokenize exactly as llama.cpp does for each of the three tokenizers (Qwen2 BPE,
Llama 3 BPE, TinyLlama SentencePiece), and 5/5 special-token cases and the end-of-generation token sets match too.

**F16 models** are held to a strict gate: logit cosine ≥ 0.999 at every position, identical argmax except at
near-ties (top-2 gap < 1e-3), and greedy decoding may diverge only at a near-tie. Minimum cosine is 0.99996 or better
on all three models, and greedy matches 64/64 on all but one Qwen prompt, which diverges at a near-tie where an fp64
reference agrees with nibble.

**Quantized models** compare against llama.cpp's *generic* (non-SIMD, no weight repacking) build with an F32 KV cache
and flash attention off, because that is the algorithm nibble reproduces; the quantization itself (block formats,
quantizers, dequantizers, block dot products) is bit-identical to ggml's generic code, pinned by golden vectors. The
gate is looser because quantizing the activations at every matmul amplifies rounding-level differences: independent
correct implementations (nibble, generic llama.cpp, an fp64 oracle) agree only to about 0.997 cosine at the worst
positions. The gate is cosine ≥ 0.995 at every position and on the prefill last row, and top-1 agreement on ≥ 90% of
positions (near-ties count as agreement); greedy decoding is reported but not asserted. Two exceptions, both recorded
in the test source with the evidence:

* **Llama-3.2 k-quant files** use a per-position floor of 0.97: on them llama.cpp's own generic and native builds
  disagree by more than 0.995 at the same positions where nibble does (the lowest nibble value is 0.9765).
* **Other k-quant files** may have one position between 0.99 and 0.995. Two files have exactly one (Qwen Q4_K_M "code",
  TinyLlama Q4_K_M "chat"); it disappears when nibble is built with `-DNIBBLE_SIMD=OFF`, so it comes from sub-ulp
  differences in the SIMD float kernels flipping one requantization rounding, not from the quantized kernels.

Minimum logit cosine over the positions of each prompt, and the lowest top-1 agreement over the three prompts:

| model | capital | code | chat | min top-1 agreement |
|---|---|---|---|---|
| Qwen2.5-0.5B F16 | 1.0000 | 1.0000 | 1.0000 | 100% |
| Qwen2.5-0.5B Q8_0 | 0.9993 | 0.9989 | 0.9973 | 99% |
| Qwen2.5-0.5B Q4_0 | 0.9992 | 0.9990 | 0.9979 | 100% |
| Qwen2.5-0.5B Q4_K_M | 0.9989 | 0.9941 | 0.9969 | 97% |
| Qwen2.5-0.5B Q5_K_M | 0.9985 | 0.9979 | 0.9954 | 97% |
| Qwen2.5-0.5B Q6_K | 0.9992 | 0.9973 | 0.9965 | 99% |
| Llama-3.2-1B F16 | 1.0000 | 1.0000 | 1.0000 | 100% |
| Llama-3.2-1B Q4_K_M | 0.9987 | 0.9907 | 0.9836 | 91% |
| Llama-3.2-1B Q5_K_M | 0.9965 | 0.9939 | 0.9765 | 98% |
| Llama-3.2-1B Q6_K | 0.9937 | 0.9959 | 0.9828 | 94% |
| TinyLlama-1.1B F16 | 1.0000 | 1.0000 | 1.0000 | 100% |
| TinyLlama-1.1B Q4_K_M | 0.9990 | 0.9990 | 0.9934 | 99% |
| TinyLlama-1.1B Q5_K_M | 0.9992 | 0.9989 | 0.9960 | 99% |
| TinyLlama-1.1B Q6_K | 0.9992 | 0.9986 | 0.9958 | 96% |

(The numbers move with rounding-level changes; they are re-recorded when kernels change. The k-quant rows were recorded
at the end of M7.)

## Performance

> The Q4_K_M and Q6_K files of all three models were benchmarked at the end of M7, with the same protocol as the
> M6 files. The Q5_0, Q5_1 and k-quant kernels are AVX2 row kernels that are bit-identical to the scalar code. They
> have no tiled prefill GEMM yet, so on those files nibble's prompt processing (`pp512`) is 1.3–4.7× slower than
> llama.cpp's, while decode is within about 2× (see the table). A prefill tile is the next performance step: Qwen2.5-0.5B Q4_K_M reaches 90.6 pp512 tok/s at t = 8 against
> 492.6 for Q4_0 (0.18×), well under the 0.5× the milestone plan set as the trigger for building one.

All numbers come from `bench/results/*.jsonl`; the charts and the tables below are
generated from them by `bench/plot.py` (`bench/plot.py` writes `docs/bench/*.svg`,
`bench/plot.py --markdown` prints the tables). `tools/bench_final.sh` runs both
tools and writes the rows.

**Machine:** Intel Core i7-1265U (2 P-cores with Hyper-Threading plus 8 E-cores, 12
logical CPUs; AVX2, FMA, F16C, AVX-VNNI), under WSL2 on a shared laptop. g++ 13
Release build; llama.cpp b11715 (`tools/bench_llamacpp.sh`).

**Why paired ratios.** Absolute tok/s on this machine drift between sessions far more
than the load average shows; clock speed, temperature and background work are not
in it. Two single-session runs at t = 8, both with load average below 2, put llama.cpp
at 1.31× nibble's Qwen2.5-0.5B Q4_0 decode speed and then at 0.69× an hour later
(the first session alternated which tool ran first per (model, t), which put nibble
first in every t = 8 pair; the second ran llama.cpp first). The second session's rows
are kept, unused by the charts and tables, in `bench/results/session2-discarded.jsonl`.
So the comparison uses
ratios from paired blocks, which cancel slow drift, and absolute tok/s are only
shown per block or per session.

**Protocol** (`tools/bench_final.sh --abba`): `nibble bench -p 512 -n 128 -r 5`
and `llama-bench` with the same sizes and thread count, each reporting the median
of 5 reps after 1 warm-up. One ABBA block runs llama.cpp, nibble, nibble, llama.cpp
back to back for one (model, thread count). Before every run the driver waits 45 s,
then until the 1-minute load average is below 2 (at most 10 minutes). The block
ratio is the mean of the two llama.cpp medians divided by the mean of the two nibble
medians; above 1 means llama.cpp is faster. Each run measures pp512 and tg128, so one
block yields both. Blocks are interleaved: every (model, t) gets its first block
before any gets its second. Qwen2.5-0.5B Q4_0 has 3 blocks at t = 1 and t = 8, the
other models 2 each. The tables give the median block ratio and its range. All 184
runs (nibble at commits `39ec442` for the M6 files and `2a0ab2c` for the k-quant files, whose engine code
is identical) started with load average below 2 except four, whose blocks are marked † and drawn hollow.

**Target (tg128 on Qwen2.5-0.5B Q4_0 within 1.5× of llama.cpp): met at t = 8, missed
at t = 1.** At t = 8, the default thread count, llama.cpp ÷ nibble is 1.19× (range
0.99–1.20× over 3 blocks). At t = 1 it is 1.68× (1.57–1.71×): every block is above
1.5×. nibble's single-thread Q4_0 decode is the gap. On pp512 nibble is faster at
t = 1 (0.74×) and slower at t = 8 (1.12×). On the other models nibble's tg128 is
level or faster at both thread counts (0.68–1.04×). Its pp512 is about 2× faster on
Q8_0 and about 2× slower on F16 Qwen at t = 8.

Paired ABBA blocks (llama.cpp ÷ nibble; tok/s are medians of the per-block means):

| model | test | t | blocks | llama.cpp ÷ nibble, median | range | nibble tok/s | llama.cpp tok/s |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | tg128 | 1 | 3 | 1.68× | 1.57–1.71× | 25.3 | 41.9 |
| Qwen2.5-0.5B Q4_0 | tg128 | 8 | 3 | 1.19× | 0.99–1.20× | 88.8 | 106.9 |
| Qwen2.5-0.5B Q4_0 | pp512 | 1 | 3 | 0.74× | 0.68–0.74× | 225.0 | 165.9 |
| Qwen2.5-0.5B Q4_0 | pp512 | 8 | 3 | 1.12× | 1.10–1.14× | 492.6 | 551.0 |
| Qwen2.5-0.5B Q8_0 | tg128 | 1 | 2 | 1.04× | 0.99–1.08× | 19.2 | 20.0 |
| Qwen2.5-0.5B Q8_0 | tg128 | 8 | 2 | 0.86× | 0.74–0.99× | 56.0 | 48.4 |
| Qwen2.5-0.5B Q8_0 | pp512 | 1 | 2 | 0.48× | 0.46–0.51× | 230.9 | 112.0 |
| Qwen2.5-0.5B Q8_0 | pp512 | 8 | 2 | 0.51× | 0.48–0.55× | 499.8 | 256.8 |
| Qwen2.5-0.5B F16 | tg128 | 1 | 2 | 0.86× | 0.84–0.87× | 14.2 | 12.2 |
| Qwen2.5-0.5B F16 | tg128 | 8 | 2 | 0.83× | 0.83–0.84× | 23.3 | 19.4 |
| Qwen2.5-0.5B F16 | pp512 | 1 | 2 | 1.06× | 1.03–1.10× | 95.0 | 100.8 |
| Qwen2.5-0.5B F16 | pp512 | 8 | 2 | 1.93× | 1.89–1.98× | 163.4 | 315.8 |
| Llama-3.2-1B F16 | tg128 | 1 | 2 | 0.68× | 0.64–0.72× | 7.9 | 5.4 |
| Llama-3.2-1B F16 | tg128 | 8 | 2 | 0.73× | 0.67–0.78× | 11.2 | 8.0 |
| Llama-3.2-1B F16 | pp512 | 1 | 2 | 1.05× | 0.99–1.11× | 33.3 | 34.9 |
| Llama-3.2-1B F16 | pp512 | 8 | 2 | 0.93× | 0.85–1.00× | 63.7 | 58.6 |
| TinyLlama-1.1B F16 | tg128 | 1 | 2 | 0.70× | 0.70–0.70× | 8.4 | 5.8 |
| TinyLlama-1.1B F16 | tg128 | 8 | 2 | 0.70× | 0.63–0.78× | 12.9 | 8.9 |
| TinyLlama-1.1B F16 | pp512 | 1 | 2 | 1.02× | 1.01–1.04× | 33.6 | 34.4 |
| TinyLlama-1.1B F16 | pp512 | 8 | 2 | 0.89× | 0.79–0.99× | 65.5 | 57.3 |
| Qwen2.5-0.5B Q4_K_M | tg128 | 1 | 2† | 1.07× | 1.05–1.09× | 20.9 | 22.3 |
| Qwen2.5-0.5B Q4_K_M | tg128 | 8 | 2 | 0.89× | 0.82–0.96× | 33.9 | 30.2 |
| Qwen2.5-0.5B Q4_K_M | pp512 | 1 | 2† | 2.02× | 1.76–2.28× | 39.4 | 79.4 |
| Qwen2.5-0.5B Q4_K_M | pp512 | 8 | 2 | 1.34× | 1.04–1.63× | 90.6 | 121.1 |
| Llama-3.2-1B Q4_K_M | tg128 | 1 | 2 | 1.86× | 1.72–2.00× | 7.6 | 14.0 |
| Llama-3.2-1B Q4_K_M | tg128 | 8 | 2 | 1.84× | 1.30–2.38× | 13.5 | 22.7 |
| Llama-3.2-1B Q4_K_M | pp512 | 1 | 2 | 4.16× | 3.66–4.67× | 13.5 | 55.2 |
| Llama-3.2-1B Q4_K_M | pp512 | 8 | 2 | 3.82× | 2.40–5.24× | 32.4 | 117.7 |
| TinyLlama-1.1B Q4_K_M | tg128 | 1 | 2 | 1.98× | 1.76–2.21× | 9.0 | 17.7 |
| TinyLlama-1.1B Q4_K_M | tg128 | 8 | 2 | 1.00× | 0.96–1.03× | 23.5 | 23.5 |
| TinyLlama-1.1B Q4_K_M | pp512 | 1 | 2 | 4.00× | 3.39–4.60× | 14.0 | 54.6 |
| TinyLlama-1.1B Q4_K_M | pp512 | 8 | 2 | 2.34× | 1.97–2.71× | 39.9 | 93.3 |
| Qwen2.5-0.5B Q6_K | tg128 | 1 | 2 | 1.14× | 1.13–1.15× | 19.7 | 22.5 |
| Qwen2.5-0.5B Q6_K | tg128 | 8 | 2 | 1.02× | 0.95–1.08× | 50.7 | 51.3 |
| Qwen2.5-0.5B Q6_K | pp512 | 1 | 2 | 1.36× | 1.33–1.38× | 88.9 | 120.7 |
| Qwen2.5-0.5B Q6_K | pp512 | 8 | 2 | 1.43× | 1.37–1.49× | 231.5 | 331.0 |
| Llama-3.2-1B Q6_K | tg128 | 1 | 2† | 1.55× | 1.38–1.72× | 6.2 | 9.6 |
| Llama-3.2-1B Q6_K | tg128 | 8 | 2 | 0.87× | 0.68–1.05× | 10.9 | 9.8 |
| Llama-3.2-1B Q6_K | pp512 | 1 | 2† | 4.67× | 4.44–4.90× | 12.0 | 56.0 |
| Llama-3.2-1B Q6_K | pp512 | 8 | 2 | 2.59× | 2.37–2.80× | 24.5 | 63.9 |
| TinyLlama-1.1B Q6_K | tg128 | 1 | 2† | 1.44× | 1.23–1.65× | 7.2 | 10.5 |
| TinyLlama-1.1B Q6_K | tg128 | 8 | 2† | 1.22× | 1.00–1.44× | 13.3 | 15.6 |
| TinyLlama-1.1B Q6_K | pp512 | 1 | 2† | 4.29× | 4.16–4.41× | 12.0 | 51.7 |
| TinyLlama-1.1B Q6_K | pp512 | 8 | 2† | 3.79× | 2.56–5.01× | 28.4 | 105.0 |

Thread sweep, Qwen2.5-0.5B Q4_0, tok/s: single session (2026-10-19 11:57–13:00 UTC, nibble 2ca56ef), used only to choose the default thread count:

| t | nibble tg128 | llama.cpp tg128 | nibble pp512 | llama.cpp pp512 |
|---|---|---|---|---|
| 1 | 19.3 | 27.4 | 143.6 | 93.6 |
| 2 | 27.4 | 47.3 | 216.3 | 149.0 |
| 4 | 39.5 | 41.6 | 272.7 | 189.5 |
| 6 | 36.8 | 43.3 | 243.2 | 221.2 |
| 8 | 40.8 | 53.5 | 290.5 | 231.3 |
| 10 | 30.5 | 37.3 | 322.1 | 189.8 |
| 12 | 31.0 | 40.5 | 321.1 | 260.0 |

† noisy: 1-minute load average > 2 when a run started (for blocks: in any of its runs).

![llama.cpp ÷ nibble ratios](docs/bench/vs-llamacpp.svg)

**Roofline** (single session, nibble `2ca56ef`, the one the thread sweep comes from). Decode reads every
weight once per token, so `model bytes × tg128 tok/s` is the weight-read bandwidth
it achieves. The GGUF file size stands in for the weight bytes; for TinyLlama that
includes an untied `token_embd` (~131 MB) that decode does not read in full.
`roofline.svg` shows it as a percentage of the `nibble bench --mem` probe: the
highest probe across that session's models at the same thread count. That probe is
a plain streaming sum and does not reach the machine's peak. In that session Q8_0
and F16 Qwen decode slightly exceed it (106% and 107%), so they are memory-bound.
Q4_0 reached 36% there (40.8 tok/s at t = 8). The ABBA blocks measured about 88.8
tok/s for it at t = 8, so that session's figure likely understates how close Q4_0
decode gets to the memory bound.

![weight-read bandwidth](docs/bench/roofline.svg)

**Ladder.** `ladder.svg` shows nibble's best decode tok/s after each M6 step
(`bench/results/m6-*.jsonl`). Each step is a single session and keeps its own thread
counts and load average. All steps except the last used reduced sizes (tg16, labelled under each step) on a
busier machine, so they show the trend, not exact speed-ups. The hand-recorded
thread-pool dispatch timings in `m6-4b-dispatch.jsonl` are not plotted.

![M6 ladder](docs/bench/ladder.svg)

**Default thread count.** With `n_threads = 0` (and no `-t` for `run` and
`chat`) nibble reads `/sys/devices/system/cpu/cpu*/cpu_capacity`. If the
capacities differ (a hybrid CPU), it uses the CPUs above the midpoint of the lowest and
highest capacity (so a prime core or ITMT-favoured cores do not shrink the big tier),
at most `hardware_concurrency`. Otherwise it uses
2/3 of `hardware_concurrency`, which is 8 here: the best nibble tg128 thread count in
the single-session sweep above (t = 4 was within noise of it). WSL2 reports a capacity of 1024 on all 12 CPUs, and a
topology of 6 cores with 2 threads each, which is not the real 2 P + 8 E. So on
this machine the rule always takes the 2/3 fallback. The hybrid branch has not
been measured on native Linux. `nibble bench -t` still defaults to 1.

## Roadmap

Done: M4 Q8_0/Q4_0, sampler, chat · M5 Llama 3.2 and TinyLlama · M6 SIMD
kernels, threading and benchmarks · M7 Q5_0/Q5_1 and the k-quants (Q4_K, Q5_K, Q6_K) with AVX2 kernels.
Next: a tiled prefill GEMM for Q5_0 and the k-quants; then Phase 2: CUDA backend.

## Documentation and contributing

* [ARCHITECTURE.md](ARCHITECTURE.md): components, data flow, the numerical-fidelity design, testing strategy.
* [CONTRIBUTING.md](CONTRIBUTING.md): build variants, checks (format, lint, types, tests), commit and PR conventions.
* [SECURITY.md](SECURITY.md): reporting vulnerabilities (nibble parses untrusted model files).

## License

MIT; see [LICENSE](LICENSE).
