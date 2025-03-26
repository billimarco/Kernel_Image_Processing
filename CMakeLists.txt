cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)

# Nome del progetto
project(Kernel_Image_Processing VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Imposta i flag di compilazione per Debug e Release
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include include/nlohmann include/stb)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89)

# Source files
set(SOURCES src/main.cpp src/kernel.cu)

# Aggiungi l'eseguibile
add_executable(${PROJECT_NAME} ${SOURCES})

# Supporto a separazione compilazione host/device
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(DEFINED CUDAToolkitDir)
    list(APPEND CMAKE_PREFIX_PATH "${CUDAToolkitDir}")
    message(STATUS "Using custom CUDA Toolkit directory: ${CUDAToolkitDir}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")  # Debug symbols
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")  # Ottimizzazione
endif()

file(COPY ${CMAKE_SOURCE_DIR}/settings DESTINATION ${CMAKE_BINARY_DIR})

# Stampa il tipo di build (Debug o Release)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

