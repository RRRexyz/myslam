cmake_minimum_required(VERSION 3.28)
project(myslam LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

if(APPLE)
    include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)
elseif(UNIX AND NOT APPLE)
    include_directories(/usr/include/eigen3)
else()
    message(FATAL_ERROR "Unsupported platform.")
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
