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
target_include_directories(nibble_tests PRIVATE include src)
target_compile_definitions(nibble_tests PRIVATE NIBBLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(nibble_tests PRIVATE nibble Catch2::Catch2WithMain)

catch_discover_tests(nibble_tests TEST_SPEC "~[e2e]")
catch_discover_tests(nibble_tests TEST_SPEC "[e2e]" EXTRA_ARGS --allow-running-no-tests TEST_SUFFIX " [e2e]" PROPERTIES LABELS e2e)

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
add_test(NAME cli_info_missing_exit COMMAND nibble info /nonexistent.gguf)
set_tests_properties(cli_info_missing_exit PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_info_missing_msg COMMAND nibble info /nonexistent.gguf)
set_tests_properties(cli_info_missing_msg PROPERTIES PASS_REGULAR_EXPRESSION "error: .*nonexistent")
add_test(NAME cli_usage COMMAND nibble frobnicate)
set_tests_properties(cli_usage PROPERTIES WILL_FAIL TRUE)
