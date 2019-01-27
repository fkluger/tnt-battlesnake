cmake_minimum_required(VERSION 2.8.9)
project(snake)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set (DRAW_SNAKE OFF)
find_package(Threads)

#Bring the headers, such as Student.h into the project
include_directories(include)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB sources "src/*.cpp")

if(NOT DRAW_SNAKE)
    list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/Draw.cpp)
endif(NOT DRAW_SNAKE)
add_executable(snake ${sources})
target_link_libraries( snake ${CMAKE_THREAD_LIBS_INIT} )
