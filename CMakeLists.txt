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
catch_discover_tests(nibble_tests TEST_SPEC "[e2e]" EXTRA_ARGS --allow-running-no-tests TEST_SUFFIX " [e2e]" PROPERTIES LABELS e2e)

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

add_test(NAME cli_info_small COMMAND nibble info ${CMAKE_BINARY_DIR}/tests/data/small.gguf)
set_tests_properties(cli_info_small PROPERTIES PASS_REGULAR_EXPRESSION
  "general.architecture = qwen2.*token_embd.weight +F16 +\\[64, 10\\]")
# Metadata strings (scalar and array elements) are printed with control bytes escaped.
string(ASCII 27 NIBBLE_ESC)
add_test(NAME cli_info_escapes_control COMMAND nibble info ${CMAKE_BINARY_DIR}/tests/data/escape.gguf)
set_tests_properties(cli_info_escapes_control PROPERTIES
  PASS_REGULAR_EXPRESSION "evil = \\\\x1b\\[31mred\\\\n.*\"\\\\x1b\\[0m\""
  FAIL_REGULAR_EXPRESSION "${NIBBLE_ESC}")
add_test(NAME cli_info_missing_exit COMMAND nibble info /nonexistent.gguf)
set_tests_properties(cli_info_missing_exit PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_info_missing_msg COMMAND nibble info /nonexistent.gguf)
set_tests_properties(cli_info_missing_msg PROPERTIES PASS_REGULAR_EXPRESSION "error: .*nonexistent")
add_test(NAME cli_usage COMMAND nibble frobnicate)
set_tests_properties(cli_usage PROPERTIES WILL_FAIL TRUE)

set(NIBBLE_TINY ${CMAKE_BINARY_DIR}/tests/data/tiny.gguf)
add_test(NAME cli_run_tiny COMMAND nibble run -m ${NIBBLE_TINY} -p hello -n 4)
set_tests_properties(cli_run_tiny PROPERTIES PASS_REGULAR_EXPRESSION "\\[max_tokens\\]")
add_test(NAME cli_run_no_prompt COMMAND nibble run -m ${NIBBLE_TINY})
set_tests_properties(cli_run_no_prompt PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_run_no_prompt_exit2 COMMAND nibble run -m ${NIBBLE_TINY})
set_tests_properties(cli_run_no_prompt_exit2 PROPERTIES PASS_REGULAR_EXPRESSION "usage: nibble run")
add_test(NAME cli_run_bad_model COMMAND nibble run -m /nonexistent -p hi)
set_tests_properties(cli_run_bad_model PROPERTIES WILL_FAIL TRUE)
# Skipped under sanitizers: ASan aborts on a multi-TB allocation instead of letting new throw.
if(NOT NIBBLE_SANITIZE)
add_test(NAME cli_run_huge_ctx COMMAND nibble run -m ${NIBBLE_TINY} -p hello -c 1000000000000)
set_tests_properties(cli_run_huge_ctx PROPERTIES PASS_REGULAR_EXPRESSION "error: .*n_ctx")
add_test(NAME cli_run_huge_ctx_exit1
  COMMAND sh -c "\"$0\" run -m \"$1\" -p hello -c 1000000000000 >/dev/null 2>&1; test $? -eq 1"
          $<TARGET_FILE:nibble_cli> ${NIBBLE_TINY})
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
  COMMAND sh -c "ulimit -v 300000; err=$(\"$0\" info \"$1\" 2>&1 >/dev/null); rc=$?; \
if [ $rc -eq 0 ]; then exit 0; fi; \
if [ $rc -eq 1 ] && printf '%s' \"$err\" | grep -q '^error:' && \
   ! printf '%s' \"$err\" | grep -q 'bad_alloc'; then exit 0; fi; \
echo \"unexpected exit $rc: $err\"; exit 1"
          $<TARGET_FILE:nibble_cli> ${NIBBLE_BIG_U8})
set_tests_properties(cli_info_big_u8_ulimit PROPERTIES FIXTURES_REQUIRED big_u8)
add_test(NAME cli_info_big_u8_clean COMMAND ${CMAKE_COMMAND} -E rm -f ${NIBBLE_BIG_U8})
set_tests_properties(cli_info_big_u8_clean PROPERTIES FIXTURES_CLEANUP big_u8)
endif()
add_test(NAME cli_run_bad_number COMMAND nibble run -m ${NIBBLE_TINY} -p hi -n abc)
set_tests_properties(cli_run_bad_number PROPERTIES PASS_REGULAR_EXPRESSION "usage: nibble run")
