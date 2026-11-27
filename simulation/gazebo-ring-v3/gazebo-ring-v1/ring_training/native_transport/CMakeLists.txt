cmake_minimum_required(VERSION 3.16)
project(ring_gazebo_transport LANGUAGES CXX)

find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_executable(ring_gazebo_transport main.cc)
target_compile_features(ring_gazebo_transport PRIVATE cxx_std_17)
target_link_libraries(ring_gazebo_transport PRIVATE gz-transport13::core gz-msgs10::core)
