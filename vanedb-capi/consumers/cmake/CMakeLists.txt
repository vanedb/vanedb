# A consumer that knows nothing about this repository: it finds the package
# from an extracted archive and links the acceptance program against both
# imported targets. Configure with the archive root on CMAKE_PREFIX_PATH:
#
#   cmake -S consumers/cmake -B build -DCMAKE_PREFIX_PATH=/path/to/vanedb-capi-<version>-<platform>
#   cmake --build build --config Release
#   ctest --test-dir build --build-config Release --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(vanedb_cmake_consumer LANGUAGES C)

find_package(vanedb REQUIRED CONFIG)
message(STATUS "vanedb ${vanedb_VERSION}, ABI ${vanedb_ABI_VERSION}")

enable_testing()
set(acceptance_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/acceptance.c")

foreach(flavour shared static)
  add_executable(acceptance_${flavour} "${acceptance_source}")
  set_target_properties(acceptance_${flavour} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  target_link_libraries(acceptance_${flavour} PRIVATE vanedb::${flavour})
  add_test(NAME acceptance_${flavour} COMMAND acceptance_${flavour} "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

if(WIN32)
  # The DLL is found next to the executable, not through an rpath.
  add_custom_command(TARGET acceptance_shared POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:vanedb::shared> $<TARGET_FILE_DIR:acceptance_shared>)
endif()
