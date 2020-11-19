cmake_minimum_required(VERSION 3.2)

add_executable(unit_test unit_test.cc)
target_link_libraries(unit_test libGBP)
target_include_directories(unit_test PUBLIC include third_party)
