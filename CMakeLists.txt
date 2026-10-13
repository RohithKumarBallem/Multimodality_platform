cmake_minimum_required(VERSION 3.20)
project(multimodal_world_models VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MMWM_BUILD_TESTS "Build C++ tests" ON)
option(MMWM_BUILD_BENCH "Build benchmarks" ON)
option(MMWM_WERROR "Treat warnings as errors" OFF)
option(MMWM_USE_BLAS "Enable the vendor-BLAS execution stage (Accelerate, or cblas if found)" ON)
option(MMWM_NATIVE "Compile for the build host's CPU (-march=native); not portable" OFF)
set(MMWM_SANITIZE "none" CACHE STRING "none|address|undefined|thread|address,undefined")

add_library(mmwm_flags INTERFACE)
target_compile_options(mmwm_flags INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
if(MMWM_WERROR)
  target_compile_options(mmwm_flags INTERFACE -Werror)
endif()
if(NOT MMWM_SANITIZE STREQUAL "none")
  target_compile_options(mmwm_flags INTERFACE
    -fsanitize=${MMWM_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
  target_link_options(mmwm_flags INTERFACE -fsanitize=${MMWM_SANITIZE})
endif()
if(MMWM_NATIVE)
  target_compile_options(mmwm_flags INTERFACE -march=native)
endif()
find_package(Threads REQUIRED)

# Optional BLAS: macOS Accelerate, otherwise any BLAS that ships a cblas.h.
set(MMWM_BLAS_DEFS "")
set(MMWM_BLAS_LIBS "")
set(MMWM_BLAS_INCLUDES "")
if(MMWM_USE_BLAS)
  if(APPLE)
    find_library(MMWM_ACCELERATE Accelerate)
    if(MMWM_ACCELERATE)
      set(MMWM_BLAS_DEFS MMWM_HAVE_BLAS MMWM_BLAS_ACCELERATE)
      set(MMWM_BLAS_LIBS ${MMWM_ACCELERATE})
    endif()
  else()
    find_package(BLAS QUIET)
    find_path(MMWM_CBLAS_INCLUDE cblas.h PATH_SUFFIXES openblas openblas-base)
    if(BLAS_FOUND AND MMWM_CBLAS_INCLUDE)
      set(MMWM_BLAS_DEFS MMWM_HAVE_BLAS)
      set(MMWM_BLAS_LIBS ${BLAS_LIBRARIES})
      set(MMWM_BLAS_INCLUDES ${MMWM_CBLAS_INCLUDE})
    endif()
  endif()
endif()
if(MMWM_BLAS_DEFS)
  message(STATUS "BLAS stage: enabled (${MMWM_BLAS_LIBS})")
else()
  message(STATUS "BLAS stage: disabled")
endif()

add_library(mmwm_core STATIC
  cpp/src/arena.cc
  cpp/src/thread_pool.cc
  cpp/src/profiler.cc
  cpp/src/exec_context.cc
  cpp/src/world_model.cc
  cpp/src/synthetic_world.cc
  cpp/src/checkpoint.cc
  cpp/src/trainer.cc)
target_include_directories(mmwm_core PUBLIC cpp/include)
target_link_libraries(mmwm_core PUBLIC Threads::Threads ${MMWM_BLAS_LIBS} PRIVATE mmwm_flags)
if(MMWM_BLAS_DEFS)
  target_compile_definitions(mmwm_core PUBLIC ${MMWM_BLAS_DEFS})
  if(MMWM_BLAS_INCLUDES)
    target_include_directories(mmwm_core PUBLIC ${MMWM_BLAS_INCLUDES})
  endif()
endif()

# Shared library with the C ABI, loaded from Python via ctypes.
add_library(mmwm SHARED cpp/src/c_api.cc)
target_link_libraries(mmwm PRIVATE mmwm_core mmwm_flags)
set_target_properties(mmwm PROPERTIES OUTPUT_NAME mmwm)

if(MMWM_BUILD_TESTS)
  enable_testing()
  foreach(t test_runtime test_model)
    add_executable(${t} cpp/tests/${t}.cc)
    target_link_libraries(${t} PRIVATE mmwm_core mmwm_flags)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

if(MMWM_BUILD_BENCH)
  add_executable(mmwm_bench_latency cpp/bench/bench_latency.cc)
  target_link_libraries(mmwm_bench_latency PRIVATE mmwm_core mmwm_flags)
endif()
