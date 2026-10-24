# Build definition for nibble.
#
# Targets:
# - nibble (static library): everything under src/, the engine itself.
# - nibble_cli (executable `nibble`): the command-line tool in cli/.
# - nibble_tests, nibble_alloc_tests: Catch2 unit/integration tests; end-to-end cases carry the `e2e` label.
# - fuzz_gguf, fuzz_load, make_seed (NIBBLE_FUZZ=ON, clang only): libFuzzer targets and their seed generator.
#
# Options: NIBBLE_NATIVE (-march=native), NIBBLE_SIMD (use the AVX2 kernels when the target supports them),
# NIBBLE_SANITIZE (value for -fsanitize=, e.g. address,undefined or thread), NIBBLE_FUZZ.
#
# Typical use: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir
# build -LE e2e
cmake_minimum_required(VERSION 3.25)
project(nibble CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options (see the header comment).
option(NIBBLE_NATIVE "Build with -march=native" ON)
option(NIBBLE_SIMD "Enable hand-written SIMD kernels when the target supports them" ON)
set(NIBBLE_SANITIZE "" CACHE STRING "Value for -fsanitize=")

# nibble_flags carries the compile settings every target shares: strict warnings as errors, -march=native, the
# SIMD switch and the sanitizer flags.
add_library(nibble_flags INTERFACE)
target_compile_options(nibble_flags INTERFACE -Wall -Wextra -Wpedantic -Werror)
if(NIBBLE_NATIVE)
  target_compile_options(nibble_flags INTERFACE -march=native)
endif()
if(NIBBLE_SIMD)
  target_compile_definitions(nibble_flags INTERFACE NIBBLE_SIMD_ENABLED=1)
endif()
if(NIBBLE_SANITIZE)
  set(NIBBLE_SANITIZE_FLAGS ${NIBBLE_SANITIZE})
  # -fsanitize=undefined leaves out float-to-int casts of NaN/out-of-range values; quantizers need them checked.
  if(NIBBLE_SANITIZE MATCHES "undefined")
    string(APPEND NIBBLE_SANITIZE_FLAGS ",float-cast-overflow")
  endif()
  # Fatal: without this UBSan prints "runtime error:" and the test still passes.
  target_compile_options(nibble_flags INTERFACE -fsanitize=${NIBBLE_SANITIZE_FLAGS} -fno-sanitize-recover=all -fno-omit-frame-pointer)
  target_link_options(nibble_flags INTERFACE -fsanitize=${NIBBLE_SANITIZE_FLAGS} -fno-sanitize-recover=all)
endif()

# The library sources are everything under src/ (globbed, re-globbed automatically when files are added).
find_package(Threads REQUIRED)
file(GLOB_RECURSE NIBBLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
# The quant dot products are pinned bitwise to ggml goldens; clang contracts a*b+c
# into FMA by default and g++ does too (GCC 13 emits vfmadd for a dot loop at -O3 -march=native),
# which changes the rounding.
# ref_ops.cpp is the scalar oracle for the SIMD kernels and must stay uncontracted too, and the AVX2
# quant kernels reproduce the scalar dots bit-for-bit, so they must not contract either. The tiled GEMM
# reproduces the row kernels (quant_avx2.cpp, matmul_f.cpp) bit-for-bit, so all three share the setting.
# ops.cpp (SIMD rmsnorm/silu/attention) needs it too: g++ contracts _mm256_mul_ps/_add_ps (and _pd) pairs,
# e.g. exp_ps's final multiply into a softmax sum, and rmsnorm must keep the scalar's per-element math.
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/quant.cpp ${CMAKE_SOURCE_DIR}/src/backend/cpu/ref_ops.cpp
  ${CMAKE_SOURCE_DIR}/src/backend/cpu/simd/quant_avx2.cpp ${CMAKE_SOURCE_DIR}/src/backend/cpu/simd/kquant_avx2.cpp
  ${CMAKE_SOURCE_DIR}/src/backend/cpu/simd/matmul_f.cpp
  ${CMAKE_SOURCE_DIR}/src/backend/cpu/simd/gemm.cpp ${CMAKE_SOURCE_DIR}/src/backend/cpu/simd/ops.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
# The engine library; its public include directory is include/ (the API), src/ stays private.
add_library(nibble STATIC ${NIBBLE_SOURCES})
target_include_directories(nibble PUBLIC include PRIVATE src)
target_link_libraries(nibble PUBLIC nibble_flags Threads::Threads)

# Catch2 is fetched at configure time (pinned tag).
include(FetchContent)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.16.1)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Tests: every tests/test_*.cpp goes into nibble_tests. The first discovery registers all non-e2e cases, the
# second the end-to-end cases with a label, a long timeout, and tolerance for builds where the models are
# missing.
enable_testing()
file(GLOB NIBBLE_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
add_executable(nibble_tests ${NIBBLE_TEST_SOURCES})
target_include_directories(nibble_tests PRIVATE include src cli)
target_compile_definitions(nibble_tests PRIVATE NIBBLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(nibble_tests PRIVATE nibble Catch2::Catch2WithMain)

catch_discover_tests(nibble_tests TEST_SPEC "~[e2e]")
catch_discover_tests(nibble_tests TEST_SPEC "[e2e]" EXTRA_ARGS --allow-running-no-tests TEST_SUFFIX " [e2e]" PROPERTIES LABELS e2e TIMEOUT 7200)

# Separate binary: replaces global operator new/delete with a counting hook.
add_executable(nibble_alloc_tests tests/alloc_tests.cpp)
target_include_directories(nibble_alloc_tests PRIVATE include src tests)
target_link_libraries(nibble_alloc_tests PRIVATE nibble Catch2::Catch2WithMain)
catch_discover_tests(nibble_alloc_tests)

# Fuzz targets: each compiles the library sources directly with the fuzzer + ASan + UBSan flags.
option(NIBBLE_FUZZ "Build libFuzzer targets (requires clang)" OFF)
if(NIBBLE_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "NIBBLE_FUZZ requires clang (CXX=clang++)")
  endif()
  set(NIBBLE_FUZZ_FLAGS -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_executable(fuzz_gguf tests/fuzz/fuzz_gguf.cpp ${NIBBLE_SOURCES})
  target_include_directories(fuzz_gguf PRIVATE include src)
  target_compile_options(fuzz_gguf PRIVATE -Wall -Wextra -Wpedantic -Werror ${NIBBLE_FUZZ_FLAGS})
  target_link_options(fuzz_gguf PRIVATE ${NIBBLE_FUZZ_FLAGS})
  target_link_libraries(fuzz_gguf PRIVATE Threads::Threads)

  add_executable(fuzz_load tests/fuzz/fuzz_load.cpp ${NIBBLE_SOURCES})
  target_include_directories(fuzz_load PRIVATE include src)
  target_compile_options(fuzz_load PRIVATE -Wall -Wextra -Wpedantic -Werror ${NIBBLE_FUZZ_FLAGS})
  target_link_options(fuzz_load PRIVATE ${NIBBLE_FUZZ_FLAGS})
  target_link_libraries(fuzz_load PRIVATE Threads::Threads)

  add_executable(make_seed tests/fuzz/make_seed.cpp ${NIBBLE_SOURCES})
  target_include_directories(make_seed PRIVATE include src tests)
  target_compile_options(make_seed PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(make_seed PRIVATE Threads::Threads)
endif()

# CLI
# The CLI. nibble_git.h (the build's git SHA, with a -dirty marker) is generated on every build by
# tools/gen_git_header.cmake and only rewritten when its content changes.
file(GLOB NIBBLE_CLI_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cli/*.cpp)
add_executable(nibble_cli ${NIBBLE_CLI_SOURCES})
set_target_properties(nibble_cli PROPERTIES OUTPUT_NAME nibble)
target_include_directories(nibble_cli PRIVATE cli src)
add_custom_target(nibble_git_header ALL
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/generated/nibble_git.h
          -P ${CMAKE_SOURCE_DIR}/tools/gen_git_header.cmake
  BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/nibble_git.h
  COMMENT "Generating nibble_git.h")
add_dependencies(nibble_cli nibble_git_header)
target_include_directories(nibble_cli PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(nibble_cli PRIVATE nibble)
# make_test_gguf generates the GGUF fixtures the CLI tests read; a post-build step runs it into
# <build>/tests/data.
add_executable(make_test_gguf tests/make_test_gguf.cpp)
target_include_directories(make_test_gguf PRIVATE include src tests)
target_link_libraries(make_test_gguf PRIVATE nibble)
add_custom_command(TARGET make_test_gguf POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/data
  COMMAND make_test_gguf ${CMAKE_BINARY_DIR}/tests/data
  COMMENT "Generating test GGUF fixtures")

# CLI tests go through tests/cli_expect.cmake: exact exit code, optional output regex.
set(NIBBLE_EXE $<TARGET_FILE:nibble_cli>)
# Declares a CLI test: runs the nibble executable through tests/cli_expect.cmake and checks the exit code, an
# output regex, a forbidden regex, stdin input and (SAME_STDOUT) determinism across two runs. Arguments go
# through one -D option, so ';' inside them is escaped.
function(nibble_cli_test name expect_exit)
  cmake_parse_arguments(T "SAME_STDOUT" "REGEX;FORBID_REGEX;STDIN" "ARGS;LABELS" ${ARGN})
  string(REPLACE ";" "$<SEMICOLON>" args "${T_ARGS}")  # keep the list in one -D argument
  set(defs -DCMD=${NIBBLE_EXE} "-DARGS=${args}" -DEXPECT_EXIT=${expect_exit})
  # Regexes are passed as direct arguments, not through the `defs` list: CMake list
  # expansion mangles unbalanced brackets.
  string(REPLACE ";" "$<SEMICOLON>" regex "${T_REGEX}")
  string(REPLACE ";" "$<SEMICOLON>" forbid "${T_FORBID_REGEX}")
  if(T_STDIN)
    list(APPEND defs -DSTDIN_FILE=${T_STDIN})
  endif()
  if(T_SAME_STDOUT)
    list(APPEND defs -DSAME_STDOUT=ON)
  endif()
  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} ${defs} "-DEXPECT_REGEX=${regex}" "-DFORBID_REGEX=${forbid}" -P ${CMAKE_SOURCE_DIR}/tests/cli_expect.cmake)
endfunction()
# Paths of the generated fixtures.
set(NIBBLE_DATA ${CMAKE_BINARY_DIR}/tests/data)
set(NIBBLE_TINY ${NIBBLE_DATA}/tiny.gguf)
set(NIBBLE_TINY_CHAT ${NIBBLE_DATA}/tiny_chatml.gguf)
set(NIBBLE_CLI_DIR ${CMAKE_SOURCE_DIR}/tests/cli)

# `nibble info` tests: output content, control-byte escaping, error cases, k-quant type names, kernel-set
# line.
nibble_cli_test(cli_info_small 0 ARGS info ${NIBBLE_DATA}/small.gguf
  REGEX "general.architecture = qwen2.*token_embd.weight +F16 +\\[64, 10\\]")
# Metadata strings (scalar and array elements) are printed with control bytes escaped.
nibble_cli_test(cli_info_escapes_control 0 ARGS info ${NIBBLE_DATA}/escape.gguf
  FORBID_REGEX "%ESC%"
  REGEX "evil = \\\\x1b\\[31mred\\\\n.*\"\\\\x1b\\[0m\"")
nibble_cli_test(cli_info_missing 1 ARGS info /nonexistent.gguf REGEX "error: .*nonexistent")
nibble_cli_test(cli_info_q6k 0 ARGS info ${NIBBLE_DATA}/q6k.gguf REGEX "w +Q6_K +\\[256\\]")
nibble_cli_test(cli_usage 2 ARGS frobnicate REGEX "unknown subcommand")

# `nibble run` tests: success, usage errors for every kind of bad flag, deterministic output with temperature
# 0 or a fixed seed.
nibble_cli_test(cli_run_tiny 0 ARGS run -m ${NIBBLE_TINY} -p hello -n 4 --temp 0 REGEX "\\[max_tokens\\]")
nibble_cli_test(cli_run_no_prompt 2 ARGS run -m ${NIBBLE_TINY} REGEX "usage: nibble run")
nibble_cli_test(cli_run_bad_model 1 ARGS run -m /nonexistent -p hi REGEX "error: ")
nibble_cli_test(cli_run_bad_number 2 ARGS run -m ${NIBBLE_TINY} -p hi -n abc REGEX "usage: nibble run")
nibble_cli_test(cli_run_temp0_deterministic 0 SAME_STDOUT
  ARGS run -m ${NIBBLE_TINY} -p hello -n 8 --temp 0)
nibble_cli_test(cli_run_seed_deterministic 0 SAME_STDOUT
  ARGS run -m ${NIBBLE_TINY} -p hello -n 8 --seed 7)
nibble_cli_test(cli_run_bad_top_p 2 ARGS run -m ${NIBBLE_TINY} -p hi --top-p -1 REGEX "usage: nibble run")
nibble_cli_test(cli_run_bad_top_p_high 2 ARGS run -m ${NIBBLE_TINY} -p hi --top-p 1.5)
nibble_cli_test(cli_run_bad_temp 2 ARGS run -m ${NIBBLE_TINY} -p hi --temp -0.5)
nibble_cli_test(cli_run_bad_top_k 2 ARGS run -m ${NIBBLE_TINY} -p hi --top-k -3)
nibble_cli_test(cli_run_bad_seed 2 ARGS run -m ${NIBBLE_TINY} -p hi --seed xyz)
nibble_cli_test(cli_run_temp_no_value 2 ARGS run -m ${NIBBLE_TINY} -p hi --temp)

# `nibble chat` tests: two turns answered, blank lines and CRLF skipped, a final line without newline, context
# exhaustion.
nibble_cli_test(cli_chat_two_turns 0 STDIN ${NIBBLE_CLI_DIR}/chat_two_turns.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok.*prompt: [0-9]+ tok")
nibble_cli_test(cli_chat_blank_lines_skipped 0 STDIN ${NIBBLE_CLI_DIR}/chat_blank_lines.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok"
  FORBID_REGEX "prompt: [0-9]+ tok.*prompt: [0-9]+ tok")  # one message -> exactly one reply
nibble_cli_test(cli_chat_crlf_blank_skipped 0 STDIN ${NIBBLE_CLI_DIR}/chat_crlf.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok"
  FORBID_REGEX "prompt: [0-9]+ tok.*prompt: [0-9]+ tok")  # CRLF blank line is blank: one reply
nibble_cli_test(cli_chat_no_trailing_newline 0 STDIN ${NIBBLE_CLI_DIR}/chat_no_newline.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok")
# The default system prompt alone takes ~90 tokens: a 100-token context fills during the first reply.
nibble_cli_test(cli_chat_context_full 0 STDIN ${NIBBLE_CLI_DIR}/chat_two_turns.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -c 100 -n 100 --temp 0 REGEX "\\[context full\\]")
# bench: JSONL schema check, bad flag values, injected load average.
add_test(NAME cli_bench_jsonl COMMAND ${CMAKE_COMMAND} -DCMD=${NIBBLE_EXE}
  "-DARGS=bench$<SEMICOLON>-m$<SEMICOLON>${NIBBLE_TINY}$<SEMICOLON>-t$<SEMICOLON>1$<SEMICOLON>-p$<SEMICOLON>8$<SEMICOLON>-n$<SEMICOLON>4$<SEMICOLON>-r$<SEMICOLON>2$<SEMICOLON>--out$<SEMICOLON>${CMAKE_BINARY_DIR}/bench_test.jsonl"
  -DOUT=${CMAKE_BINARY_DIR}/bench_test.jsonl -DN_LINES=2 -P ${CMAKE_SOURCE_DIR}/tests/bench_jsonl_check.cmake)
add_test(NAME cli_bench_git_sha COMMAND ${CMAKE_COMMAND} -DCMD=${NIBBLE_EXE} -DSRC=${CMAKE_SOURCE_DIR}
  -P ${CMAKE_SOURCE_DIR}/tests/git_sha_check.cmake)
add_test(NAME cli_bench_quote_model COMMAND ${CMAKE_COMMAND} -DCMD=${NIBBLE_EXE} -DTINY=${NIBBLE_TINY}
  -DDIR=${CMAKE_BINARY_DIR} -P ${CMAKE_SOURCE_DIR}/tests/bench_quote_check.cmake)
nibble_cli_test(cli_bench_bad_threads 2 ARGS bench -m ${NIBBLE_TINY} -t 1,x -p 8 -n 4 -r 1 REGEX "usage: nibble bench")
nibble_cli_test(cli_bench_no_model 2 ARGS bench REGEX "usage: nibble bench")
nibble_cli_test(cli_bench_mem 0 ARGS bench -m ${NIBBLE_TINY} -t 1 -p 8 -n 4 -r 1 --mem REGEX "membw")
add_test(NAME cli_bench_load_warning COMMAND ${CMAKE_COMMAND} -E env NIBBLE_LOADAVG_PATH=${NIBBLE_CLI_DIR}/loadavg_high.txt
  ${CMAKE_COMMAND} -DCMD=${NIBBLE_EXE} "-DARGS=bench$<SEMICOLON>-m$<SEMICOLON>${NIBBLE_TINY}$<SEMICOLON>-t$<SEMICOLON>1$<SEMICOLON>-p$<SEMICOLON>8$<SEMICOLON>-n$<SEMICOLON>4$<SEMICOLON>-r$<SEMICOLON>1"
  -DEXPECT_EXIT=0 "-DEXPECT_REGEX=warning: load average 3.5" -P ${CMAKE_SOURCE_DIR}/tests/cli_expect.cmake)
# bench/plot.py on a fixture: ABBA block ratios (incomplete block ignored, noisy run flagged), sweep rows
# (newest duplicate wins), model-less rows and session2-discarded.jsonl (999 tok/s rows) skipped.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME bench_plot_markdown COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/bench/plot.py
    --results ${CMAKE_SOURCE_DIR}/tests/data/plot --markdown)
  set_tests_properties(bench_plot_markdown PROPERTIES PASS_REGULAR_EXPRESSION
    "tg128 \\| 8 \\| 2† \\| 1.25× \\| 1.00–1.50× \\| 45.0 \\| 55.0 \\|.*\\| 2 \\| 30.0 \\| 40.0 \\|.*\\| 8 \\| 50.0† \\|")
  add_test(NAME bench_plot_svg COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/bench/plot.py
    --results ${CMAKE_SOURCE_DIR}/tests/data/plot --out ${CMAKE_BINARY_DIR}/plot_test)
endif()
nibble_cli_test(cli_chat_no_model 2 ARGS chat REGEX "usage: nibble chat")
nibble_cli_test(cli_chat_bad_model 1 ARGS chat -m /nonexistent REGEX "error: ")
nibble_cli_test(cli_chat_bad_temp 2 ARGS chat -m ${NIBBLE_TINY_CHAT} --temp abc)
nibble_cli_test(cli_chat_unknown_flag 2 ARGS chat -m ${NIBBLE_TINY_CHAT} --bogus 1)

# Skipped under sanitizers: ASan aborts on a multi-TB allocation instead of letting new throw.
if(NOT NIBBLE_SANITIZE)
nibble_cli_test(cli_run_huge_ctx 1 ARGS run -m ${NIBBLE_TINY} -p hello -c 1000000000000 REGEX "error: .*n_ctx")
endif()
# A 50 MB u8-array GGUF must not amplify into hundreds of MB: nibble info under a
# ~300 MB address-space limit exits 0 (or 1 with a translated "error:", not a raw
# std::bad_alloc), never aborts.
# Skipped under sanitizers: ASan reserves far more virtual memory than the limit.
if(NOT NIBBLE_SANITIZE)
set(NIBBLE_BIG_U8 ${CMAKE_BINARY_DIR}/tests/data/big_u8.tmp.gguf)
add_test(NAME cli_info_big_u8_gen COMMAND make_test_gguf --big-u8 ${NIBBLE_BIG_U8} 50000000)
set_tests_properties(cli_info_big_u8_gen PROPERTIES FIXTURES_SETUP big_u8)
add_test(NAME cli_info_big_u8_ulimit
  COMMAND sh -c "ulimit -v 300000; exec \"$0\" -DCMD=\"$1\" \"-DARGS=info;$2\" -DEXPECT_EXIT=0 -P \"$3\""
          ${CMAKE_COMMAND} ${NIBBLE_EXE} ${NIBBLE_BIG_U8} ${CMAKE_SOURCE_DIR}/tests/cli_expect.cmake)
set_tests_properties(cli_info_big_u8_ulimit PROPERTIES FIXTURES_REQUIRED big_u8)
add_test(NAME cli_info_big_u8_clean COMMAND ${CMAKE_COMMAND} -E rm -f ${NIBBLE_BIG_U8})
set_tests_properties(cli_info_big_u8_clean PROPERTIES FIXTURES_CLEANUP big_u8)
endif()

nibble_cli_test(cli_info_kernels 0 ARGS info ${NIBBLE_DATA}/small.gguf REGEX "kernels: (avx2\\+fma\\+f16c(\\+avxvnni)?|scalar)\n")
