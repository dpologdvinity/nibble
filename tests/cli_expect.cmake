# Run a CLI command and check its exit code (and optionally output).
#   cmake -DCMD=<exe> [-DARGS="a;b"] -DEXPECT_EXIT=<n> [-DEXPECT_REGEX=<re>]
#         [-DFORBID_REGEX=<re> ('%ESC%' = ESC byte)]
#         [-DSTDIN_FILE=<path>] [-DSAME_STDOUT=ON] -P cli_expect.cmake
# A signal-killed child yields a non-numeric RESULT ("Segmentation fault", ...), which
# never equals a numeric EXPECT_EXIT, so crashes fail.
foreach(v CMD EXPECT_EXIT)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "cli_expect: ${v} not set")
  endif()
endforeach()

function(run_once out_var all_var)
  set(extra)
  if(STDIN_FILE)
    set(extra INPUT_FILE ${STDIN_FILE})
  endif()
  execute_process(COMMAND ${CMD} ${ARGS} ${extra}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT "${rc}" STREQUAL "${EXPECT_EXIT}")
    message(FATAL_ERROR "expected exit '${EXPECT_EXIT}', got '${rc}'\n--- stdout ---\n${out}\n--- stderr ---\n${err}")
  endif()
  set(${out_var} "${out}" PARENT_SCOPE)
  set(${all_var} "${out}${err}" PARENT_SCOPE)
endfunction()

run_once(out1 all1)
if(NOT "${EXPECT_REGEX}" STREQUAL "" AND NOT "${all1}" MATCHES "${EXPECT_REGEX}")
  message(FATAL_ERROR "output does not match '${EXPECT_REGEX}'\n--- stdout+stderr ---\n${all1}")
endif()
if(NOT "${FORBID_REGEX}" STREQUAL "")
  string(ASCII 27 ESC)
  string(REPLACE "%ESC%" "${ESC}" forbid "${FORBID_REGEX}")  # raw ESC via -D is fragile
  if("${all1}" MATCHES "${forbid}")
    message(FATAL_ERROR "output matches forbidden '${FORBID_REGEX}'\n--- stdout+stderr ---\n${all1}")
  endif()
endif()
if(SAME_STDOUT)
  string(LENGTH "${out1}" out1_len)
  if(out1_len EQUAL 0)
    message(FATAL_ERROR "stdout is empty; determinism check would be vacuous")
  endif()
  run_once(out2 all2)
  if(NOT "${out1}" STREQUAL "${out2}")
    message(FATAL_ERROR "stdout differs between runs\n--- run 1 ---\n${out1}\n--- run 2 ---\n${out2}")
  endif()
endif()
