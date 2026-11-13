# Configures nibble with tests/fake_cxx_no_vnni_intrinsic.sh as the compiler and checks that CMake notices the
# toolchain cannot build the AVX-VNNI tier and leaves it out, instead of failing later while compiling it.
# Run as: cmake -DSRC=<source dir> -DWORK=<scratch dir> -DWRAPPER=<script> -P vnni_probe_check.cmake
file(REMOVE_RECURSE ${WORK})
execute_process(
  COMMAND ${CMAKE_COMMAND} -S ${SRC} -B ${WORK} -DCMAKE_BUILD_TYPE=Release -DNIBBLE_BUILD_TESTS=OFF
          -DCMAKE_CXX_COMPILER=${WRAPPER}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "configure failed (${rc}):\n${out}\n${err}")
endif()
if(NOT out MATCHES "nibble: AVX-VNNI tier not built \\(the toolchain cannot compile the AVX-VNNI intrinsic")
  message(FATAL_ERROR "the VNNI tier was not skipped for a toolchain that cannot encode vpdpbusd:\n${out}")
endif()
file(REMOVE_RECURSE ${WORK})
