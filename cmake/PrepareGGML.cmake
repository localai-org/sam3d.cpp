include_guard(GLOBAL)

# Derived from skin-tokens.cpp's build-copy patch phase. Content-addressed
# copies avoid deleting any existing checkout or build directory on reconfigure.
function(sam3d_prepare_ggml result)
  find_package(Git 2.20 REQUIRED)
  set(source "${PROJECT_SOURCE_DIR}/ggml")
  set(expected "e91ded11bdcd78c42f9c8d3978ff6686eb4c1226")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" rev-parse HEAD
    RESULT_VARIABLE rc OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT rc EQUAL 0 OR NOT revision STREQUAL expected)
    message(FATAL_ERROR "Experimental GGML patches require initialized upstream ${expected}")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" status --porcelain --untracked-files=all
    COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE dirty)
  if(dirty)
    message(FATAL_ERROR "GGML submodule must be pristine; patches only modify a build-tree copy")
  endif()
  file(GLOB patches CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/patches/ggml/*.patch")
  list(SORT patches)
  if(NOT patches)
    message(FATAL_ERROR "No bundled GGML patches")
  endif()
  file(SHA256 "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" preparer_hash)
  set(material "${expected}:${preparer_hash}")
  foreach(patch IN LISTS patches)
    file(SHA256 "${patch}" hash)
    string(APPEND material ":${hash}")
  endforeach()
  string(SHA256 fingerprint "${material}")
  set(destination "${CMAKE_BINARY_DIR}/_deps/ggml-source-${fingerprint}")
  if(NOT EXISTS "${destination}/.sam3d-patch-complete")
    if(EXISTS "${destination}")
      message(FATAL_ERROR "Incomplete patched source ${destination}; inspect/remove that exact build artifact before retrying")
    endif()
    file(MAKE_DIRECTORY "${destination}")
    file(COPY "${source}/" DESTINATION "${destination}" PATTERN ".git" EXCLUDE)
    foreach(patch IN LISTS patches)
      execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
        "${GIT_EXECUTABLE}" -C "${destination}" apply --check --recount "${patch}"
        COMMAND_ERROR_IS_FATAL ANY)
      execute_process(COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
        "${GIT_EXECUTABLE}" -C "${destination}" apply --recount "${patch}"
        COMMAND_ERROR_IS_FATAL ANY)
    endforeach()
    # Do not let upstream's version probe accidentally describe our parent repo.
    file(WRITE "${destination}/.git" "gitdir: ${destination}/.no-git\n")
    file(WRITE "${destination}/.sam3d-patch-complete" "${fingerprint}\n")
  endif()
  message(STATUS "Using build-copy GGML patches: ${fingerprint}")
  set(${result} "${destination}" PARENT_SCOPE)
endfunction()
