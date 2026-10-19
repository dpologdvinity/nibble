# cmake -DSRC=<repo> -DOUT=<header> -P gen_git_header.cmake
# Writes OUT with NIBBLE_GIT_SHA (short sha, "-dirty" suffix if tracked files differ,
# "unknown" outside a repo); leaves OUT untouched when the content is unchanged.
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${SRC}
  OUTPUT_VARIABLE sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
if(NOT rc EQUAL 0 OR NOT sha)
  set(sha unknown)
else()
  execute_process(COMMAND git diff --quiet HEAD WORKING_DIRECTORY ${SRC} RESULT_VARIABLE drc ERROR_QUIET)
  if(NOT drc EQUAL 0)
    set(sha "${sha}-dirty")
  endif()
endif()
set(content "#pragma once\n#define NIBBLE_GIT_SHA \"${sha}\"\n")
set(old "")
if(EXISTS ${OUT})
  file(READ ${OUT} old)
endif()
if(NOT old STREQUAL content)
  file(WRITE ${OUT} "${content}")
endif()
