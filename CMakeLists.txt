cmake_minimum_required(VERSION 3.2)
project(curplsh)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to DEBUG.")
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

option(BuildTest "BuildTest" ON)
option(BuildBench "BuildBench" OFF)
option(BuildShared "BuildShared" OFF)

if (BuildShared)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

find_package(CUDA 9.0 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

if (BuildTest)
  enable_testing()
  add_subdirectory(test)
endif()

if (BuildBench)
  add_subdirectory(bench)
endif()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMake generator      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type         : ${LIB_TYPE}")
message(STATUS "  Build tests          : ${BuildTest}")
message(STATUS "  Build benchmarks     : ${BuildBench}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "")
