add_library(body_motion body_motion.cpp)

add_library(camera camera.cpp)
target_link_libraries(camera ${OpenCV_LIBS})

add_executable(myslam main.cpp)