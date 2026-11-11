# SDD ledger — plan: docs/superpowers/plans/2026-10-17-nibble-m4.md

Spec: docs/superpowers/specs/2026-10-17-nibble-design.md (reachable)
Branch: feat/m4 from master fdfdde6

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 → T2 | blocks, quantize_row_q8_0, dots | consistent |
| T2 → T3 | Q matmul/embed; reserve(ReserveParams) | consistent |
| T2 ↔ T5/T6 | all modify tests/tiny_model.h (Q dims, ChatML tokens, arch variant) | sequential; ok |
| T4 → T5 | Sampler(SamplerParams, n_vocab), sample(span) | consistent |
| T5 → T6 | PromptPart, append, n_past(), ContextFull on overflow | T5 keeps throw for fresh (append=false) too-long prompt; T6 first turn uses append=false → would throw instead of ContextFull → R1 |
| T5 | escape_control moves cli/escape.h → src/util/escape.h | test_cli_escape.cpp + cmd_info must follow; ok |
| T5 | GenParams default temperature 0.8 | existing greedy tests must set temperature=0; CLI tests check only markers; ok |
| T6 → T7 | Chat API; non-qwen2 tiny variant needed by CLI test | make_test_gguf must emit it; ok |
| T7 | cli_expect.cmake replaces WILL_FAIL/regex tests | consistent |
| T3 | e2e over 3 models | runtime ~3× (≈20 min); acceptable |

## Rulings

- Ruling R1: the `generate(std::span<const PromptPart>, …)` overload returns ContextFull (no throw) when the new tokens don't fit, for both append=false and append=true; the string overload keeps the M0–M3 throw for back-compat. Chat therefore never throws on a long first turn — cost if wrong: one behaviour difference between overloads, documented in nibble.h.

## Progress
- Ruling R2: accept Task 1 round-trip bound amax/254 + amax·2^-10 (plan's amax/254·1.01 ignored fp16 rounding of d: error ≤ d/2 + 127·d·2^-11) — plan error — cost if wrong: test slightly looser.
Task 1: implementer DONE (fdfdde6..5d3623f). In review.
Task 1: minor (deferred): F32/F16 dequantize_row skips n<0 check
Task 1: minor (deferred): no tests for bad n / unsupported dtype / roundf tie / multi-block unaligned row
Task 1: minor (deferred): FMA contraction may differ bitwise from ggml (tolerance absorbs) — consider -ffp-contract=off in M6
Task 1: complete (commits fdfdde6..5d3623f, review clean)
Task 2: implementer DONE_WITH_CONCERNS (5d3623f..6b03f9d). tiny Q8 cos .999875, Q4 .999528 (close to bound). In review.
Task 2: minor (deferred): Weights::bind dtype-rejection branch unreachable via GGUF, untested (needs stub backend)
Task 2: minor (deferred): Transformer reports any reserve failure as "attention scratch … -c"
Task 2: minor (deferred): Q4_0 cosine WARN noise → INFO
Task 2: minor (deferred): no Transformer-level test that quant scratch doesn't grow across Q prefill/decode
Task 2: minor (deferred): dead TinyModelOpts::q4_tensor
Task 2: minor (deferred): naive oracle shares dequantize_row with SUT
Task 2: complete (commits 5d3623f..6b03f9d, review clean)
Task 3: BLOCKED — q8_0 min cos 0.993672 (chat), q4_0 0.997761; non-tie argmax mismatches. Quantizer/dots match ggml generic; float-vs-double alone moves cos 0.998–0.9999 (requantization amplifies rounding).
- Ruling R3: quantized goldens come from a second llama.cpp build on the generic CPU path (GGML_NATIVE=OFF, AVX/AVX2/FMA/F16C/AVX512 OFF, GGML_LLAMAFILE=OFF, GGML_CPU_REPACK=OFF) with F32 KV cache (llama_ref --kv-f32) — that build executes the reference algorithm nibble implements; thresholds unchanged; f16 goldens untouched — cost if wrong: one more ~10 min llama.cpp build; if nibble still fails, it's a real bug.
Task 3: R3 follow-up still BLOCKED — vs generic llama.cpp + F32 KV: q8_0 min cos .99672, q4_0 .99658; three-way (nibble / generic llama / fp64 oracle with requant) all disagree at 1e-3–2.5e-3 → intrinsic noise floor. Asking user for quantized thresholds.
- User decision (2026-10-17): quantized e2e gate = cosine ≥ 0.995 at every position (incl. batch-prefill last row) AND top-1 agreement (equal or near-tie <1e-3 either side) at ≥ 90% of positions per prompt; greedy matches reported, not asserted. F16 gate unchanged (0.999, near-tie, greedy asserted). Generic-reference goldens (R3) kept.
Task 3: implementer DONE (6b03f9d..5132b92). e2e 11/11 under user gate. q8 min cos .99672, q4 .996577; top-1 agreement ≥97%; greedy q8 capital 13/64 (gap .038), q4 capital 7/64 (gap .031). In review.
Task 3: review — Important: README lacks --generic prerequisite and quantized-gate description. Minors 1–3 (empty-array under set -u, missing generic binary hint, make_quants partial file) folded into fix round (cheap, scripts touched anyway); minor 4 (comma one-liner) deferred.
Task 3: fix round 1/5 (4 addressed, 0 open — README generic+gate, KV set -u, generic hint, .tmp mv; commits 5132b92..6389649)
Task 3: minor (deferred): make_quants .tmp not removed on failure (no trap)
Task 3: minor (deferred): README "tighter gate not attainable" overstates; F16 preface scope ambiguous
Task 3: minor (deferred): llama_ref comma one-liner readability
Task 3: complete (commits 6b03f9d..6389649, review clean)
Task 4: implementer DONE (6389649..137e3a3). freq .66728. No-alloc test in nibble_alloc_tests binary. In review.
Task 4: review — Important: NaN logits make comparator non-strict-weak → UB in nth_element/sort. Minor ±inf → NaN probs → arbitrary pick (folded into fix round, same code).
Task 4: minor (deferred): to_string(float) 6-decimal messages; test gaps (top-k tie by index, u≈1 fallback, size mismatch throw, top_k+top_p combo); n_vocab > INT32_MAX unchecked
User-requested code review (Tasks 1–2, fdfdde6..6b03f9d): With fixes. Important: (1) dot_q8_0_q8_0 uses (s*dw)*da vs ggml generic s*(dw*da) — 38.6% rows differ in last bits; plan formula was the Q4_0 form (plan error); (2) no bit-exact golden-vector tests for quantize_row_q8_0_ref / dots (k+0.5 ties, fp16(d)≠d). Noise-floor conclusion: agreed. Suggested falsification: fix (1), build NIBBLE_NATIVE=OFF + -ffp-contract=off, re-run quantized e2e.
CORRECTION: earlier ledger statement "Quantizer/dots match ggml generic" was false for the Q8_0 dot.
Queued after Task 4 fix round: "Task 2b" fix dispatch for review items (1)+(2) + falsification experiment.
Review minors (deferred): xq_.resize bad_alloc past reserve; no-growth test only n=1; h_ re-quantized 5× per layer (M6 perf); include order.
Task 4: fix round 1/5 (2 addressed, 0 open — NaN total order, ±inf handling; commits 137e3a3..d2692e5)
Task 4: minor (deferred): fallback idx[keep-1] may return zero-prob -inf/NaN token on 1e-16 rounding shortfall with top_p=1
Task 4: complete (commits 6389649..d2692e5, review clean)
Task 2b: dispatched (bit-exact Q8_0 dot, golden-vector quant tests, -ffp-contract=off falsification experiment), base d2692e5
Task 2b: implementer DONE (d2692e5..896f10c). Q8 dot bracket fixed (only discrepancy); golden vectors bit-exact; e2e 11/11; strict build quant 6/6. Experiment: ±1e-3 cosine moves, no convergence → noise floor confirmed. build-strict q4 chat .995786 near gate. In review.
Task 2b: review Approved. Caveat recorded: bit-exact Q8 fix lowered q8_0 min cos on all 3 prompts (.999497→.999409, .998747→.998544, .99672→.996164) — consistent small shift, not random; floor dominated by something other than dot scale rounding (activation-requant flips / FMA in ref_ops / accumulator precision). Conclusion "noise floor, not bug" stands with this caveat.
Task 2b: minor (deferred): dequant / q4 dot tests don't check input PRNG hash; build-strict q4 chat margin .0008 vs .995 gate; -ffp-contract decision open for M6
Task 2b: complete (commits d2692e5..896f10c, review clean)
Task 5: implementer DONE (896f10c..16ab652). g++ 135/135, ASan 132/132, clang 133/135 (golden dot_q4_0/q8_0 bitwise fail under clang — pre-existing from Task 2b).
- Ruling R4: Task 2b defect — clang contracts quant dot into FMA; compile src/quant.cpp with -ffp-contract=off for all compilers (set_source_files_properties) so the bit-exact golden holds on g++ and clang — cost if wrong: none (quant kernels scalar until M6).
- Ruling R5: invariant "every token delivered to on_token is in the KV cache when room remains" — on MaxTokens and Cancelled the last delivered token is forwarded if n_past < n_ctx; EOS (never delivered) is not. Keeps cache == visible text for chat continuation — cost if wrong: one extra forward at stop.
- Ruling R6: append=false resets n_past to 0 even when it returns without forwarding (ContextFull / max_tokens<=0), so a fresh call never leaves stale context for a later append — cost if wrong: none.
Task 5: review — Needs fixes: R4/R5/R6 deviations (sites listed in review), R1 empty-list ambiguity.
- Ruling R7: R5 includes ContextFull (forward last delivered token when n_past < n_ctx); R1 reading = empty parts with append=true continues from cache (documented), append=false still throws "empty prompt" — cost if wrong: one API edge behaviour.
Task 5: minors folded into fix round: flaky "model is movable" (temp 0), random_device only when sampling, n_ctx>context_length check before heavy load, stale ASan SKIP. Deferred: escape tests for pre-tokenizer/weights sites.
Task 5: fix round 1/5 (all addressed — R4 -ffp-contract=off, R5/R7 cache delivered tokens, R6 reset, R1 doc, 4 minors; commits 16ab652..12c2523). Controller verified build-clang golden tests 6/6, 0 warnings.
Task 5: minor (deferred): append=false empty-prompt throw leaves cache intact (undocumented); MaxTokens equivalence test covers first qualifying prompt only
- Ruling R8: add a clang++ Release test job to CI in Task 7 (CI only ran tests under g++, which let the R4 FMA defect slip) — cost if wrong: one more CI job.
Task 5: complete (commits 896f10c..12c2523, review clean)
- Ruling R9: non-qwen2 'no chat template' is unreachable via a loadable model until M5 (config rejects other archs first). Test it via unit function chat_format_for(arch) in src/chat.h; drop Task 7's CLI 'chat on non-qwen2' test until M5 — cost if wrong: one test moves to M5.
Task 6: implementer DONE (12c2523..3e297bc). 150/150 Release+clang, 147/147 ASan. Smoke q8_0: 'The capital of France is Paris.' / 'The capital of Italy is Rome.' (14 new tokens prefilled). Concerns: Chat state desync if Session reset/throws; ChatTestAccess in public header. In review.
Task 6: review Approved (no blocking).
- Ruling R10: elevate Task 6 minors 1–2 (public API): Chat throws Error("chat context lost") when session.n_past() != expected after a started conversation; replace ChatTestAccess in nibble.h with one neutral `friend struct detail::Access;` defined only in src/test_access.h, and drop per-turn last_parts copy if a test seam replaces it. Minors 3–5 (n_prompt_tokens check in injection test, thread-safety + "construct new Chat" docs) folded in — cost if wrong: small.
Task 6: R10 follow-up DONE (3e297bc..da7eba6): expected_past guard, hook removed (option a), injection/default-system tests were silently ContextFull on 64-ctx tiny model → TinyModelOpts::context_length, now 256. In re-review.
Task 6: fix round 1/5 (R10 follow-up: 4 addressed, 0 open; commits 3e297bc..da7eba6)
Task 6: minor (deferred): desync guard compares n_past only (coincident n_past undetected); forward-throw path reasoned not fault-injected; replay tie-back is probabilistic on tiny model
Task 6: complete (commits 12c2523..da7eba6, review clean)
Task 7: implementer DONE (da7eba6..425b54f). 167/167 Release+clang, 165/165 ASan; R8 clang CI job, R9 applied. In review.
Task 7: review — Important: FAIL_REGULAR_EXPRESSION vacuous under cmake -P wrapper for cli_info_escapes_control and cli_chat_blank_lines_skipped (negative assertions lost). Minor determinism tests lack non-empty stdout check (folded in).
Task 7: minor (deferred): cmd_chat duplicates stats block; from_chars float needs libstdc++; set_property placement; no huge --top-k/-n test; document unbalanced ']' limitation
Task 7: fix round 1/5 (3 addressed, 0 open — FORBID_REGEX for escape + one-reply, non-empty stdout; commits 425b54f..0faafc7)
Task 7: complete (commits da7eba6..0faafc7, review clean)
Task 8: DONE_WITH_CONCERNS — Release 167/167, ASan 165/165, e2e 11/11, fuzz_gguf 571548 / fuzz_load 513973 runs clean. Min cos f16 .99996, q8 .99616, q4 .99658. Chat Q4_0 turn 2 "Italy is the capital of Italy" (no Rome); Q8_0 smoke in Task 6 answered Rome. Speed gen tok/s (load 18–25): f16 0.8, q4 1.4, q8 1.4. → final review to check llama.cpp greedy on the same Q4_0 two-turn prompt.
Task 8: complete (no commits)
Final review: With fixes (docs). Chat Q4_0 verdict: model quality — 140-token turn-2 cache token-exact; llama.cpp default + generic both answer "Italy is the capital of Italy…"; with -n 48 turn 1, turn 2 says Rome. Task 8 Step 2 "second mentions Rome" unmet on Q4_0 due to model quality; met on Q8_0 (Task 6).
Final fix wave scope: README stale Q8 numbers + gate wording; nibble.h validation-throw note; CRLF strip in chat; doc seeded-chat per-turn RNG restart; doc sampler order vs llama.cpp. All other minors deferred per final triage (M5/M6).
Final fix wave: DONE (0faafc7..58c7f9b) — CRLF strip + test, README Q8 .99941/.99854/.99616, docs. ASan 166 vs 168: 2 CLI tests guarded off under sanitizers by design. In re-review.
Final: merged feat/m4 → master (ff fdfdde6..58c7f9b), 168/168 non-e2e on merged tree; branch deleted; not pushed.
