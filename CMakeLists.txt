cmake_minimum_required(VERSION 3.31)
project(network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

set(SOURCES
    src/main.cpp
    src/network.cpp
    src/readcsv.cpp)

include_directories(include)

add_executable(network ${SOURCES})

target_link_libraries(network PRIVATE ${ARMADILLO_LIBRARIES})

