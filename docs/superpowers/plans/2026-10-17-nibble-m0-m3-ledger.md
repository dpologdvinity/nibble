# SDD ledger — plan: docs/superpowers/plans/2026-10-17-nibble-m0-m3.md

Spec: docs/superpowers/specs/2026-10-17-nibble-design.md (reachable)
Branch: feat/m0-m3 (created by controller on unborn master; no commits on master)

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 → all | `nibble::Error`, `ThreadPool`, CMake `nibble`/`nibble_tests`, `[e2e]` label | consistent |
| T2 → T3, T7, T9, T10 | `gguf::File`, `Tensor`, `DType`, `GgufWriter` | consistent |
| T2 ↔ T3 | `make_seed` (fuzz) vs `make_test_gguf` (CLI data) both use GgufWriter | two small tools, distinct purposes; ok |
| T4 → T7, T12 | `tokenize.txt` line format, `Golden` reader, corpus `====` separator | T7 must parse corpus exactly like llama_ref; plan names no shared parser → R2 |
| T5 → T6, T7, T10 | `utf8_decode`, categories, byte map, `Utf8Streamer` | consistent |
| T6 → T7 | `pretokenize(text, max_digit_run)` | consistent |
| T7 → T9 | `tiny_tokenizer_gguf` vocab 261 (256 bytes + 5) = T9 vocab 261 | consistent |
| T8 → T9 | Backend ops; rmsnorm/add_bias take F32 `Tensor` | consistent with T9 F32 norms/biases |
| T9 → T10, T12 | `Transformer::forward(tokens,pos0,logits_out host)`; `tiny_model_gguf(seed,wtype,zero_output_norm,eos_id)` | consistent; T10 eos test uses eos_id=0 |
| T10 → T11 | public API only | consistent |
| T3 → T11 | `cli/commands.h`, `make_test_gguf` extended with tiny.gguf | consistent |
| T1 self | Step 2 creates branch | already done by controller → R1 |
| T2 self | tests list vs Interfaces | consistent |
| T3 self | Files list vs steps | consistent after plan fix |
| T4 self | Files incl. test_golden_file.cpp | consistent |
| T5–T8 self | tests vs signatures | consistent |
| T9 self | qwen2 requires q/k/v biases; tiny model must include them | implied, carry into dispatch |
| T10 self | Global "no heap allocation in generation loop" vs `decode_token`/`Utf8Streamer` returning `std::string` | conflict → R3 |
| T11, T12 self | stop-reason names `eos/max_tokens/context_full/cancelled` | consistent |

## Rulings

- Ruling R1: Task 1 Step 2 (create branch) is a no-op; controller already created `feat/m0-m3` — avoids a failing `checkout -b` — cost if wrong: none.
- Ruling R2: Task 4 adds `std::vector<std::string> read_corpus(const std::string& path)` to `tests/golden_file.h` implementing the `====` rule; Task 7 golden test uses it — one parser for both sides — cost if wrong: small refactor.
- Ruling R3: "no heap allocation in generation loop" binds `Transformer::forward` and backend ops (all buffers preallocated); token-string handling in `Session::generate` (decode_token, Utf8Streamer) may allocate small strings — spec intent is the compute path; cost if wrong: a small perf fix in M6.

## Progress
Task 1: BLOCKED — sudo apt-get denied by permissions; awaiting user to install cmake+clang
Task 1: implementer DONE_WITH_CONCERNS (efbe445). Notes: TSan on this WSL2 kernel needs `setarch -R` (ASLR); e2e discover group uses `--allow-running-no-tests`. In review.
Task 1: minor (deferred): ThreadPool n_threads error omits value (thread_pool.cpp:123)
Task 1: minor (deferred): ctor not exception-safe if thread creation throws mid-loop (thread_pool.cpp:124)
Task 1: minor (deferred): parallel_for not reentrant; undocumented
Task 1: minor (deferred): no test that workers actually participate
Task 1: minor (deferred): include order nit thread_pool.h
Task 1: ⚠️ e2e empty-group discovery — resolved: implementer report shows `ctest -LE e2e` 5/5 pass with zero e2e tests
Task 1: complete (commits root..efbe445, review clean)
Task 2: implementer DONE (efbe445..d444d20). In review.
Task 2: review — Critical: F32 row_bytes product wraps (gguf.cpp:222), hostile tensor accepted with tiny nbytes; Minors listed below
- Ruling R4: elevate Task 2 minor "unsupported-type message lacks 'convert with llama-quantize'" into fix round — spec §6.2 gives the wording; it's spec compliance — cost if wrong: one string.
- Ruling R5: elevate Task 2 minor "scalar Bool stored as int64" into fix round — Value's bool alternative is the interface later tasks see; bool general.alignment accepted as 1 is wrong — cost if wrong: small.
Task 2: minor (deferred): reserve() amplification lets bad_alloc escape on multi-GB crafted files (gguf.cpp:166,185,108)
Task 2: minor (deferred): duplicate-key / Q4_K message tests use weak substrings
Task 2: minor (deferred): make_seed uses std::system mkdir; prefer std::filesystem
Task 2: minor (deferred): fuzz harness skips nbytes==0, can't see size-wrap class
Task 2: ⚠️ CI fuzz job assumes clang++ on ubuntu-24.04 runner — deferred to final review (CI not run locally)
Task 2: fix round 1/5 (3 addressed, 0 open — row_bytes wrap, §6.2 hint, scalar bool; commits d444d20..4a6e32b)
Task 2: complete (commits efbe445..4a6e32b, review clean)
Task 3: implementer DONE_WITH_CONCERNS (4a6e32b..cef8b4f): real qwen2.5-0.5b fp16 has 291 tensors incl. explicit output.weight F16 [896,151936]
- Ruling R6: plan's Task 3 expectation (290 tensors, no output.weight) was a plan error; official Qwen GGUF ships explicit output.weight. Code stands. Task 9's tied-embedding fallback stays (tiny model + Llama-3.2 in M5 need it) — cost if wrong: none.
Task 3: ⚠️ download re-run on complete file — resolved by controller: re-run exits 0, file intact
Task 3: minor (deferred): WILL_FAIL cli tests can't distinguish exit 2 from crash (plan-mandated)
Task 3: minor (deferred): main catches only nibble::Error; add std::exception fallback
Task 3: minor (deferred): cmd_info usage error exits 1 not 2
Task 3: minor (deferred): u64 metadata >2^63 prints negative
Task 3: minor (deferred): download writes directly to final name (no .part)
Task 3: complete (commits 4a6e32b..cef8b4f, review clean)
Task 4: implementer DONE (cef8b4f..081d032). llama.cpp pinned b11715; capital top-1 = 12095 " Paris"; corpus 37 entries; .gitattributes keeps corpus CRLF (verified blob + worktree have 2 CRs). In review.
Task 4: ⚠️ items resolved — tests glob/NIBBLE_SOURCE_DIR from Task 1; ctest -L e2e 1/1 and sizes per implementer report
Task 4: minor (deferred): llama_ref n_ubatch default 512 limits prompt length
Task 4: minor (deferred): llama_ref atoi on n_gen unchecked
Task 4: minor (deferred): corpus splitter edge forms undocumented (leading ====)
Task 4: minor (deferred): .gitattributes should also cover tests/e2e/prompts/*.txt
Task 4: minor (deferred): golden_file.h allocates before size validation
Task 4: minor (deferred): make_golden.sh doesn't check tokenize line count
Task 4: complete (commits cef8b4f..081d032, review clean)
Task 5: implementer DONE (081d032..de7cd8d), unidata 15.0.0. In review.
- Ruling R7: elevate Task 5 minor "Utf8Streamer emits invalid UTF-8 when held/incoming bytes are invalid" into fix round — spec §6.3 requires on_token to always get valid UTF-8; byte-level models can emit lone byte tokens; Task 10 asserts it — cost if wrong: small extra code.
Task 5: minor (deferred): utf8_append doesn't guard surrogates / >U+10FFFF
Task 5: minor (deferred): no tests for E0 A0 / ED 9F / F4 8F acceptance edges
Task 5: minor (deferred): unidata 15.0.0 vs llama.cpp table version — rare-codepoint divergence risk; golden corpus is the check
Task 5: fix round 1/5 (1 addressed, 0 open — streamer validity; commits de7cd8d..d5f5f36)
Task 5: minor (deferred): flush emits one U+FFFD per held prefix (maximal-subpart), not per byte — valid output
Task 5: complete (commits 081d032..d5f5f36, review clean)
Task 6: implementer DONE (d5f5f36..1c71e49). Implementer's unassigned-codepoint divergence concern REFUTED by reviewer (llama fills all in-range cps with UNDEFINED flag; flags==0 only past end). In review.
Task 6: minor (deferred): thin edge-case tests (lowercase/3-letter contractions, lone apostrophe, \r\n, tab/U+3000 prefix, combining mark, multibyte offsets, invalid utf8 via pretokenize)
Task 6: minor (deferred): rule 8 unreachable (same as llama.cpp), kept as safety net
Task 6: complete (commits d5f5f36..1c71e49, review clean)
Task 7: implementer DONE (1c71e49..79b0478). Golden 37/37. Special partition follows llama.cpp per-token longest-first passes; Unknown skipped like Control. In review.
Task 7: ⚠️ golden 37/37 — resolved by controller: `ctest -L e2e` Passed (not Skipped)
Task 7: minor (deferred): merges with >1 space accepted (unmatchable, harmless)
Task 7: minor (deferred): is_eog uses 3 strings; llama.cpp set larger — document
Task 7: minor (deferred): test gaps (leftmost tie, stale bigram, eos out of range, bad token_type, mid-string utf8, golden extra lines)
Task 7: minor (deferred): special partition O(S·n)
Task 7: complete (commits 1c71e49..79b0478, review clean)
Task 8: implementer DONE (79b0478..20e6bf0). Attention scratch grow-only with ctor hints max_ctx/max_heads. In review.
Task 8: review — Important: attention scratch grows during decode unless ctor hints passed; Task 9 wiring can't pass them
- Ruling R8: add `virtual void Backend::reserve(int64_t n_ctx, int64_t n_heads)` (CPU sizes attention scratch; CUDA may no-op); Transformer ctor (Task 9) calls it; drop CpuBackend ctor hints — ownership stays with the backend, model code needs no CPU knowledge — cost if wrong: one virtual to remove.
Task 8: minor (deferred): rmsnorm/add_bias don't check F32 dtype
Task 8: minor (deferred): attention_heads doesn't check n_heads % n_kv_heads (load-time check in Task 9)
Task 8: minor (deferred): alloc(n<0) throws bad_alloc not Error
Task 8: minor (deferred): unused <memory> include
Task 8: minor (deferred): passthrough ops not tested at backend level
Task 8: fix round 1/5 (1 addressed, 0 open — Backend::reserve; commits 20e6bf0..b3faa65)
Task 8: minor (deferred): scratch_capacity() returns size() not capacity()
Task 8: minor (deferred): attention not safe for concurrent calls on one backend (single Session per Model in v1)
- Ruling R9: ThreadPool::parallel_for takes const std::function&; lambdas capturing >16 bytes heap-allocate per op call in libstdc++, which technically violates R3 in steady state. Deferred to M6 (switch to a non-owning function_ref) — correctness unaffected — cost if wrong: small per-op malloc until M6.
Task 8: complete (commits 79b0478..b3faa65, review clean)
Task 9: implementer DONE (b3faa65..09509fa). Smoke: Qwen capital argmax 12095 ' Paris'. TDD order not followed (RED reproduced after). K/V projected directly into cache rows. In review.
Task 9: minor (deferred → carried into Task 10 dispatch): Transformer holds const Weights& / Backend& — dangling-ref risk; Task 10 must keep Impl address-stable and Session must not outlive Model
Task 9: minor (deferred): dtype error lists Q4_0 as supported while rejecting Q4_0 (weights.cpp:248)
Task 9: minor (deferred): divisibility checks untested though test named for them (test_model.cpp:569)
Task 9: minor (deferred): tied-vs-explicit "differ" assertion weak (RNG shift)
Task 9: minor (deferred): no check w.layers.size()==cfg.n_layers in Transformer ctor
Task 9: minor (deferred): bad token id in later chunk leaves earlier KV rows written; embed message lacks id
Task 9: minor (deferred): TDD order violated (RED reproduced after)
Task 9: complete (commits b3faa65..09509fa, review clean)
Task 10: implementer DONE (09509fa..1f32cc7). Smoke: " Paris. It is the largest city in Europe and the second largest city in the"
- Ruling R11: accept ContextFull when n_prompt + n_generated >= n_ctx (controller's dispatch step order was off by one vs the brief's test) — brief test is the authority; costs at most one unused KV slot — cost if wrong: one token less capacity.
Task 10: minor (deferred): n_ctx<0 checked after Weights::bind; negative n_threads silently auto
Task 10: minor (deferred): n_ctx > n_ctx_train accepted unchecked — decide clamp/error in M4 plan
Task 10: minor (deferred): after Cancelled, flush tail may trigger one more callback (doc note)
Task 10: minor (deferred): max_tokens<=0 still runs prefill
Task 10: minor (deferred): moved-from Model null impl_; moving Model with live Session undocumented
Task 10: minor (deferred): test gaps (session reuse, default n_ctx, max_tokens<=0, utf8 test asserts ≥1 piece, ofstream failure, <unistd.h>)
Task 10: complete (commits 09509fa..1f32cc7, review clean)
Task 11: implementer DONE (1f32cc7..e092c72). Manual: ' Paris. ...' prompt 3.7 tok/s, gen 2.2 tok/s (scalar baseline). gguf_writer put() changed to resize+memcpy for GCC13 false positive. In review.
Task 11: minor (deferred): cli_run_no_prompt_exit2 / bad_number use PASS_REGULAR_EXPRESSION so exit code 2 unverified; bad_model WILL_FAIL passes on any crash
Task 11: minor (deferred): gen tok/s slightly overstated (first token counted in gen, its time in prefill)
Task 11: complete (commits 1f32cc7..e092c72, review clean)
Task 12: BLOCKED — capital pos 50: golden argmax 11 (20.73255) vs nibble 13; golden gap 0.00264 > 1e-3; greedy 46/64; code/chat 64/64; min cos 0.99996. Gathering evidence (naive double oracle at pos 50) before asking user.
Task 12: evidence — naive double oracle agrees with nibble at capital pos 50 (gap −0.00088 vs nibble −0.00087; golden +0.00264); cos(naive,nibble)=1.0, cos(naive,golden)=0.99999918. Flip is llama.cpp F16 activation rounding. Awaiting user threshold decision.
- User decision (2026-10-17): near-tie = golden top-2 gap < 1e-3 OR nibble top-2 gap < 1e-3 (threshold value unchanged). Applies to per-position argmax check and greedy divergence check.
Task 12: implementer DONE (e092c72..137ebe0). e2e 5/5; min cos capital .999991 code .999978 chat .999962; greedy 46*/64*/64. Speed drop to 0.4–0.8 tok/s explained by unrelated chrome-headless load (~3.5 cores), not code. In review.
Task 12: ⚠️ timeout — resolved: e2e 401 s < ctest default 1500 s
Task 12: minor (deferred): no REQUIRE on golden n_gen/total; stray comment; top_gap rescan
Task 12: complete (commits e092c72..137ebe0, review clean)
Final review: With fixes. Important: (1) Session destroy leaks KV/scratch (~123 MB each) — no Backend::free; (2) bad_alloc/system_error abort (huge -c, crafted n_kv reserve, -t 2000), §6.2 KV message missing; (3) tokenizer n_vocab not checked vs token_embd rows.
Note: ruling numbering skips R10 (controller numbering error, no lost ruling).
Final triage: fix-before-merge = T1 ctor exception safety, T2 reserve amplification, T3 main catch, T8 alloc(n<0) — all folded into Important #2. All other minors deferred to M4–M6 plans per final review triage.
Final fix wave: DONE (137ebe0..611d8ef). In scoped re-review.
Final fix wave: re-review — all 3 addressed, no new Critical/Important. Residual minors parked for M4–M6: KV-loop catch only Error (bad_alloc leak path); no test reaches KV-specific message branch; gguf cap test non-discriminating; memset may OOM-kill under overcommit; CLI tests use `COMMAND nibble` (lib target name) instead of $<TARGET_FILE:nibble_cli>.
Final: complete (commits root..611d8ef)
