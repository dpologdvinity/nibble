# Generates tests/data/quant_golden.inc from llama.cpp's reference quant code (pinned b11715, generic
# non-SIMD ggml build). Requires third_party/build-generic (tools/build_llama_cpp.sh --generic).
#
# Regenerate:
#   cmake -S tools/quant_vectors -B build-quant-vectors
#   cmake --build build-quant-vectors
#   build-quant-vectors/quant_vectors tests/data/quant_golden.inc
cmake_minimum_required(VERSION 3.25)
project(quant_vectors CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(LLAMA_GENERIC_BIN ${ROOT}/third_party/build-generic/bin CACHE PATH "bin dir of the generic llama.cpp build")
find_library(GGML_BASE ggml-base PATHS ${LLAMA_GENERIC_BIN} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_CPU ggml-cpu PATHS ${LLAMA_GENERIC_BIN} NO_DEFAULT_PATH REQUIRED)
add_executable(quant_vectors quant_vectors.cpp)
target_compile_options(quant_vectors PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(quant_vectors PRIVATE ${ROOT}/third_party/llama.cpp/ggml/include ${ROOT}/tests)
target_link_libraries(quant_vectors PRIVATE ${GGML_CPU} ${GGML_BASE})
set_target_properties(quant_vectors PROPERTIES BUILD_RPATH ${LLAMA_GENERIC_BIN})
