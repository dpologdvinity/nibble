# cmake -DSRC=<repo> -DOUT=<header> -P gen_git_header.cmake
# Writes OUT with NIBBLE_GIT_SHA (short sha, "-dirty" suffix if tracked files differ or src,
# cli or include hold untracked files, "unknown" outside a repo); leaves OUT untouched when
# the content is unchanged.
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${SRC}
  OUTPUT_VARIABLE sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
if(DEFINED ENV{NIBBLE_GIT_SHA} AND NOT "$ENV{NIBBLE_GIT_SHA}" STREQUAL "")
  # Builds without a .git directory (container image) pass the revision in explicitly.
  set(sha "$ENV{NIBBLE_GIT_SHA}")
  if(NOT sha MATCHES "^[0-9A-Za-z._-]+$")
    message(FATAL_ERROR "NIBBLE_GIT_SHA must be a revision name (letters, digits, . _ -), got '${sha}'")
  endif()
  string(SUBSTRING "${sha}" 0 7 sha)  # `git rev-parse --short` length, so image and tarballs report the same revision
elseif(NOT rc EQUAL 0 OR NOT sha)
  set(sha unknown)
else()
  execute_process(COMMAND git diff --quiet HEAD WORKING_DIRECTORY ${SRC} RESULT_VARIABLE drc ERROR_QUIET)
  # Untracked sources are globbed into the build too.
  execute_process(COMMAND git status --porcelain --untracked-files=normal -- src cli include
    WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE untracked ERROR_QUIET)
  if(NOT drc EQUAL 0 OR untracked)
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
