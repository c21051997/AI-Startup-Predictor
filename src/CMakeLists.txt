cmake_minimum_required(VERSION 3.10)
project(StartupPredictor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include Google Test headers and libraries
include_directories(/opt/homebrew/opt/googletest/include)

# Add source files for the main project and tests
add_executable(MyTests test.cpp)

# Link Google Test libraries and pthread to the test executable
target_link_libraries(MyTests /opt/homebrew/opt/googletest/lib/libgtest.a /opt/homebrew/opt/googletest/lib/libgtest_main.a pthread)

