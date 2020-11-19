cmake_minimum_required(VERSION 3.2)

project(
  GBP VERSION 1.0
  DESCRIPTION "Loopy Gaussian Belief Propagation"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")

add_library(libGBP src/variable.cc src/factor.cc)
target_include_directories(libGBP PUBLIC include third_party)
add_executable(GBP src/main.cc)
target_link_libraries(GBP libGBP)
target_include_directories(GBP PUBLIC include third_party)

#add_subdirectory(test)
