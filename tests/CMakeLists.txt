include(CTest)

add_executable(test_body_motion test_body_motion.cpp)
target_link_libraries(test_body_motion body_motion)
add_test(NAME test_body_motion COMMAND test_body_motion)