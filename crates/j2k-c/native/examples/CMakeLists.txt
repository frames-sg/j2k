# SPDX-License-Identifier: MIT OR Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(j2k_examples LANGUAGES C CXX)
find_package(J2K CONFIG REQUIRED)

add_executable(j2k_decode decode.c)
target_compile_features(j2k_decode PRIVATE c_std_11)
target_link_libraries(j2k_decode PRIVATE J2K::j2k)

add_executable(j2k_decode_cpp decode.cpp)
target_compile_features(j2k_decode_cpp PRIVATE cxx_std_17)
target_link_libraries(j2k_decode_cpp PRIVATE J2K::j2k)

# Windows loads j2k.dll from the program's directory, so copy it there.
if(WIN32)
    foreach(example j2k_decode j2k_decode_cpp)
        add_custom_command(TARGET ${example} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E
            copy_if_different "$<TARGET_FILE:J2K::j2k>" "$<TARGET_FILE_DIR:${example}>")
    endforeach()
endif()
