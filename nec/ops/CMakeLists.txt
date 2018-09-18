cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER /usr/local/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
 
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
 
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -DNDEBUG -lfaiss -undefined dynamic_lookup ${CMAKE_CXX_FLAGS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(nearest_neighbor SHARED nearest_neighbor/add_to_index.cc nearest_neighbor/k_nearest_neighbors.cc)
