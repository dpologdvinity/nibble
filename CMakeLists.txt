cmake_minimum_required(VERSION 3.25)
project(nibble CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NIBBLE_NATIVE "Build with -march=native" ON)
set(NIBBLE_SANITIZE "" CACHE STRING "Value for -fsanitize=")

add_library(nibble_flags INTERFACE)
target_compile_options(nibble_flags INTERFACE -Wall -Wextra -Wpedantic -Werror)
if(NIBBLE_NATIVE)
  target_compile_options(nibble_flags INTERFACE -march=native)
endif()
if(NIBBLE_SANITIZE)
  target_compile_options(nibble_flags INTERFACE -fsanitize=${NIBBLE_SANITIZE} -fno-omit-frame-pointer)
  target_link_options(nibble_flags INTERFACE -fsanitize=${NIBBLE_SANITIZE})
endif()

find_package(Threads REQUIRED)
file(GLOB_RECURSE NIBBLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
# The quant dot products are pinned bitwise to ggml goldens; clang contracts a*b+c
# into FMA by default (g++ does not under -std=c++20), which changes the rounding.
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/quant.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
add_library(nibble STATIC ${NIBBLE_SOURCES})
target_include_directories(nibble PUBLIC include PRIVATE src)
target_link_libraries(nibble PUBLIC nibble_flags Threads::Threads)

include(FetchContent)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.16.1)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

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

option(NIBBLE_FUZZ "Build libFuzzer targets (requires clang)" OFF)
if(NIBBLE_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "NIBBLE_FUZZ requires clang (CXX=clang++)")
  endif()
  set(NIBBLE_FUZZ_FLAGS -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
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
file(GLOB NIBBLE_CLI_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cli/*.cpp)
add_executable(nibble_cli ${NIBBLE_CLI_SOURCES})
set_target_properties(nibble_cli PROPERTIES OUTPUT_NAME nibble)
target_include_directories(nibble_cli PRIVATE cli src)
target_link_libraries(nibble_cli PRIVATE nibble)
add_executable(make_test_gguf tests/make_test_gguf.cpp)
target_include_directories(make_test_gguf PRIVATE include src tests)
target_link_libraries(make_test_gguf PRIVATE nibble)
add_custom_command(TARGET make_test_gguf POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests/data
  COMMAND make_test_gguf ${CMAKE_BINARY_DIR}/tests/data
  COMMENT "Generating test GGUF fixtures")

# CLI tests go through tests/cli_expect.cmake: exact exit code, optional output regex.
set(NIBBLE_EXE $<TARGET_FILE:nibble_cli>)
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
set(NIBBLE_DATA ${CMAKE_BINARY_DIR}/tests/data)
set(NIBBLE_TINY ${NIBBLE_DATA}/tiny.gguf)
set(NIBBLE_TINY_CHAT ${NIBBLE_DATA}/tiny_chatml.gguf)
set(NIBBLE_CLI_DIR ${CMAKE_SOURCE_DIR}/tests/cli)

nibble_cli_test(cli_info_small 0 ARGS info ${NIBBLE_DATA}/small.gguf
  REGEX "general.architecture = qwen2.*token_embd.weight +F16 +\\[64, 10\\]")
# Metadata strings (scalar and array elements) are printed with control bytes escaped.
nibble_cli_test(cli_info_escapes_control 0 ARGS info ${NIBBLE_DATA}/escape.gguf
  FORBID_REGEX "%ESC%"
  REGEX "evil = \\\\x1b\\[31mred\\\\n.*\"\\\\x1b\\[0m\"")
nibble_cli_test(cli_info_missing 1 ARGS info /nonexistent.gguf REGEX "error: .*nonexistent")
nibble_cli_test(cli_usage 2 ARGS frobnicate REGEX "unknown subcommand")

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

nibble_cli_test(cli_chat_two_turns 0 STDIN ${NIBBLE_CLI_DIR}/chat_two_turns.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok.*prompt: [0-9]+ tok")
nibble_cli_test(cli_chat_blank_lines_skipped 0 STDIN ${NIBBLE_CLI_DIR}/chat_blank_lines.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok"
  FORBID_REGEX "prompt: [0-9]+ tok.*prompt: [0-9]+ tok")  # one message -> exactly one reply
nibble_cli_test(cli_chat_no_trailing_newline 0 STDIN ${NIBBLE_CLI_DIR}/chat_no_newline.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -n 4 --temp 0 REGEX "prompt: [0-9]+ tok")
# The default system prompt alone takes ~90 tokens: a 100-token context fills during the first reply.
nibble_cli_test(cli_chat_context_full 0 STDIN ${NIBBLE_CLI_DIR}/chat_two_turns.txt
  ARGS chat -m ${NIBBLE_TINY_CHAT} -c 100 -n 100 --temp 0 REGEX "\\[context full\\]")
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
