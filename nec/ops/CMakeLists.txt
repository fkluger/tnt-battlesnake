cmake_minimum_required(VERSION 3.5)

project(TensorflowOps)

set(CMAKE_CXX_STANDARD 11)

if(APPLE)
    set(CMAKE_C_COMPILER /usr/local/opt/llvm/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
endif()

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

set(CMAKE_CXX_FLAGS "-Wall -O3 -DNDEBUG")
set(CMAKE_SHARED_LINKER_FLAGS "-lfaiss ${CMAKE_SHARED_LINKER_FLAGS}")
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup ${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-I ${CMAKE_SOURCE_DIR} ${CMAKE_CXX_FLAGS}")
    link_directories("./faiss")
endif()

include_directories(${Tensorflow_INCLUDE_DIRS})

add_library(nearest_neighbor SHARED nearest_neighbor/add_to_index.cc nearest_neighbor/k_nearest_neighbors.cc)