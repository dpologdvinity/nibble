# The binary's baked-in SHA must match HEAD (optionally with -dirty).
execute_process(COMMAND ${CMD} bench --version OUTPUT_VARIABLE got OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${SRC}
  OUTPUT_VARIABLE want OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc ERROR_QUIET)
if(NOT rc EQUAL 0)
  set(want unknown)
endif()
if(NOT (got STREQUAL want OR got STREQUAL "${want}-dirty"))
  message(FATAL_ERROR "binary sha '${got}' != HEAD '${want}'")
endif()
