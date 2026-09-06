cmake_minimum_required(VERSION 3.20)
project(vanedb_c_example LANGUAGES C)

set(VANEDB_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
    CACHE PATH "Directory containing the built Rust C ABI library")
add_library(vanedb_runtime SHARED IMPORTED)
set_target_properties(vanedb_runtime PROPERTIES
    IMPORTED_LOCATION "${VANEDB_LIBRARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}vanedb_capi${CMAKE_SHARED_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../include")
if(WIN32)
    set_target_properties(vanedb_runtime PROPERTIES
        IMPORTED_IMPLIB "${VANEDB_LIBRARY_DIR}/vanedb_capi.dll.lib")
endif()

add_executable(quickstart quickstart.c)
set_target_properties(quickstart PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(quickstart PRIVATE vanedb_runtime)
if(WIN32)
    add_custom_command(TARGET quickstart POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:vanedb_runtime> $<TARGET_FILE_DIR:quickstart>)
endif()
enable_testing()
add_test(NAME c_quickstart COMMAND quickstart)

add_executable(acceptance ../tests/acceptance.c)
set_target_properties(acceptance PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(acceptance PRIVATE vanedb_runtime)
add_test(NAME c_acceptance COMMAND acceptance "${CMAKE_CURRENT_BINARY_DIR}")
