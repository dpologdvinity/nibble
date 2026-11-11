# SDD ledger — plan: docs/superpowers/plans/2026-10-18-nibble-m5.md

Spec: docs/superpowers/specs/2026-10-17-nibble-design.md (reachable)
Branch: feat/m5 from master 58c7f9b

## Pre-flight scan
| Pair / task | Finding |
|---|---|
| T1 → T6/T7 | Zephyr prompt strings recorded in T1 from the embedded template; T6 pins them, T7 uses chat_zephyr.txt — consistent |
| T1 → T3/T4/T7 | goldens (tokenize corpus, add_special=false; prompts, add_special=true → BOS) — corpus checks need no BOS; e2e prompt check needs T5 auto-BOS → T7 after T5, ok |
| T2 → T7 | llama arch forward; T2 smoke feeds golden tokens, independent of tokenizer — ok |
| T3/T4 → T5 | encode() becomes encode_parts wrapper in T5; T3/T4 tests via encode() keep working — ok |
| T5 → T6 | Chat uses parts + auto-BOS; markup without BOS — consistent with decision 5 |
| T6 | chat_format_for signature changes (arch, template); existing R9 unit tests must be updated, not dropped |
| T8 | touches sampler/chat/cli after T6 — sequential, ok |
| all | Qwen numbers must stay identical — each task runs existing suite |
Scan clean; no rulings needed.

## Rulings

## Progress
Task 1: implementer DONE (58c7f9b..97d4499). Metadata as expected; Zephyr prompt recorded; goldens sized ok. In review.
Task 1: minor (deferred): curl -C - 416 on complete file; .deps-ok marker ignores requirements edits; API fetch no retry; fixed Llama 3 date vs strftime_now
Task 1: complete (commits 58c7f9b..97d4499, review clean)
Task 2: implementer DONE (97d4499..edda593). Smoke llama32 cos 1.000000 top1 12366=golden; tinyllama cos 1.000000 top1 3681=golden. No biases in converted files. In review.
Task 2: review Approved with Important: llama files with attn_{q,k,v}.bias silently ignored (plan said ignore).
- Ruling R1: overrule plan Task 2 "biases ignored" — a llama-arch file carrying attn_{q,k,v}.bias throws Error naming the tensor (silent wrong logits worse than rejecting; real Llama/TinyLlama files have none) — cost if wrong: a rare derivative model won't load until supported.
Task 2: fix round 1 folds minors: rope_freqs DCHECK in run_chunk; f32_only message generic wording; qwen2 head_dim-key test; "invalid model:" prefix consistency.
Task 2: minor (deferred): no negative control for rope mode sensitivity; single TEST_CASE vs two
Task 2: fix round 1/5 (4 addressed, 0 open — reject llama biases, rope_freqs DCHECK, f32 message, invalid-model prefix + qwen2 test; commits edda593..ac6a198)
Task 2: minor (deferred): bias test covers only blk.0 q; qwen2 key test doesn't assert prefix
Task 2: complete (commits 97d4499..ac6a198, review clean)
Task 3: implementer DONE (ac6a198..55ebcdb). Llama3 golden 37/37, Qwen 37/37; ignore_merges added for llama3 pre; EOG = llama.cpp full list. In review.
Task 3: review Approved; Important: is_eog misses llama.cpp's FIM-id EOG additions (<|fim_pad|>, <|repo_name|>/<|fim_repo|>, <|file_sep|> — present in Qwen2.5); </s> newly EOG for Qwen (matches llama.cpp, but a Qwen behaviour change).
- Ruling R2: EOG set is pinned against llama.cpp itself — llama_ref gains `eog <model> <out>` (all ids where llama_vocab_is_eog), make_golden writes models/golden/<name>/eog.txt, [e2e] test requires nibble's is_eog set == llama.cpp's for every model; implement FIM additions to make it pass; Qwen EOG change accepted because it matches llama.cpp (spec: exact parity) — cost if wrong: Qwen stops on a few more tokens, exactly as llama.cpp does.
Task 3: minor folded: comment on skipped <|end|> / </s> removal rules; document narrowed pre aliases.
Task 3: fix round DONE (55ebcdb..1a83ebd): EOG == llama.cpp for 4 models; Qwen EOG +{128247,151662,151663,151664}. In re-review.
Task 3: fix round 1/5 (2 addressed, 0 open — EOG pinned to llama.cpp, docs; commits 55ebcdb..1a83ebd)
Task 3: minor (carried into Task 4): EOG test silently skips any model whose tokenizer throws (must require TinyLlama once SPM lands); FIM multi-match comment says "lowest id" but llama.cpp takes first in text order
Task 3: minor (deferred): double blank line; make_golden usage stray note
Task 3: complete (commits ac6a198..1a83ebd, review clean)
Task 4: implementer DONE_WITH_CONCERNS (1a83ebd..20e4934). TinyLlama golden 37/37 first try; Qwen/Llama3 37/37; EOG 5/5 compared, skip removed.
- Ruling R3: accept implementer's correction — llama.cpp b11715 token_to_id is unordered_map (llama-vocab.cpp:1969), so FIM multi-match order is unspecified there; nibble keeps lowest id (deterministic), comment corrected — cost if wrong: none for supported vocabs (single match).
Task 4: review Approved. Minors carried into Task 5 (tokenizer touched there): equal-score tie-break test; 1-char/UNK byte-fallback tests; reject non-finite scores at load. Deferred: dead id<0 branch + comment wording (spm.cpp:77, spm.h:131); uppercase-only byte hex; per-char string in escape.
Task 4: complete (commits 1a83ebd..20e4934, review clean)
Task 5: implementer DONE (20e4934..75ec455). Prompt tokens exact all 5 models × 3 prompts; adjacent raw parts merged; double BOS skipped (llama.cpp warns :3617/:3716); empty prompt → no BOS (still throws). In review.
Task 5: review Approved. Carried: nibble.h "as one string" doc overclaim (special tokens never span parts) → Task 6; prompt-token golden silent skip counter → Task 7. Deferred: UNK byte-fallback test comment (nibble ≠ llama.cpp there on purpose); random-split test special=false only. Note: UserDefined tokens match in user text (as llama.cpp; not Control).
Task 5: complete (commits 20e4934..75ec455, review clean)
Task 6: implementer DONE_WITH_CONCERNS (75ec455..c5b8e52). Smoke Llama3+TinyLlama: Paris / Rome. Controller ran full ASan: 192/192, 0 warnings. In review.
Task 6: review Approved. Carried to Task 7: assert chat_turn first-turn concat == chat_llama3.txt / chat_zephyr.txt (verified manually only). Deferred: long comment lines nibble.h:78-80,128; private chat_template() visible in header; Llama3 system not |trim'd + fixed date (comment); "no chat template" message when arch known but template unmatched; Zephyr can't tell empty vs absent system (documented).
Task 6: complete (commits 75ec455..c5b8e52, review clean)
Task 7: implementer DONE (c5b8e52..16ff95f). e2e 21/21 (43m48s); llama32 min cos .999996/.999981/.999990, tinyllama .999999/.999999/.999997, greedy 64/64 all; non-e2e 195/195. In review.
Task 7: minor (deferred → Task 8 cleanup): README long lines :78, :112-114; comma-operator one-liner in run_prompt
Task 7: complete (commits c5b8e52..16ff95f, review clean)
Task 8: implementer DONE (16ff95f..2ba826b, 5 commits). 201/201 Release+clang, 199/199 ASan. Seeded outputs differ from M4 (by design). In review.
Task 8: minor (deferred → final): nibble.h sentence that throwing/max_tokens<=0 direct calls also trip "chat context lost" and shift seed index; wrap src/chat.cpp:63; fmt "-nan"; pick_index "~1" doc
Task 8: complete (commits 16ff95f..2ba826b, review clean)
Task 9: DONE — Release 201/201, ASan 199/199 (2 sanitizer-guarded CLI tests by design), clang 201/201, e2e 21/21, fuzz 758k/948k clean. Min cos Qwen f16 .99996, Llama3.2 .99999, TinyLlama .999997, Q8 .99616, Q4 .99658. Chat both new models: Paris / Rome. Speed (load ~10): Llama gen 2.1, TinyLlama gen 2.2 tok/s.
Task 9: complete (no commits)
Final review: With fixes. Important: (1) EOG-by-name tokens not promoted to Control (llama-vocab.cpp:3056-3060) → Qwen `a</s>b` special parse diverges; (2) convert_hf.sh downloads unpinned resolve/main; converter can reach trust_remote_code=True via repo JSON → RCE risk + non-reproducible. Minor fix-now: nibble.h:100 wording, README:26 wording, Llama3 |trim (#5), README Qwen seed/EOG change note (#6), BPE add_bos without bos id → throw (#7), long lines, T3 blank line + make_golden usage note, T4 dead id<0 branch. Deferred to M6 per final triage: add_eos_token, rope_freqs finite check, and the rest listed in review.
Final fix wave: DONE (2ba826b..67fe72f, 7 commits). 203/203 Release+clang, 201/201 ASan, e2e 22/22 unchanged numbers, special goldens 5/5, fuzz clean. Pins: unsloth/Llama-3.2-1B-Instruct@5a8abab4, TinyLlama@fe8a4ea1; reconversion byte-identical. In re-review.
Final fix wave re-review: all 10 addressed, no new Critical/Important. Deferred to M6: Unicode |trim (ASCII-only, mark shortcut:), convert_hf extra files in models/hf unverified / GGUF skip not re-verified / pip not hash-pinned, special-golden test silent skip when golden missing, long lines in tokenizer.cpp + test_chat.cpp:511.
Final: complete (commits 58c7f9b..67fe72f), ready to merge pending user approval.
