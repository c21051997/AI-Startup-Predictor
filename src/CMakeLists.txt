cmake_minimum_required(VERSION 3.10)
project(StartupPredictor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include Google Test headers and libraries
include_directories(/opt/homebrew/opt/googletest/include)
include_directories(${CMAKE_SOURCE_DIR})  # Include the project directory for headers

# Add the source files for the main project
add_library(StartupPredictor
    LogisticRegression.cpp
    Dataset.cpp  # Add this if you have a Utils.cpp
)

# Add the test files
add_executable(MyTests
    test.cpp
)

# Link the StartupPredictor library to the test executable
target_link_libraries(MyTests
    StartupPredictor  # Link the main project library to the tests
    /opt/homebrew/opt/googletest/lib/libgtest.a
    /opt/homebrew/opt/googletest/lib/libgtest_main.a
    pthread
)

# Ensure that CMake knows where to find Google Test
target_include_directories(MyTests PRIVATE /opt/homebrew/opt/googletest/include)


