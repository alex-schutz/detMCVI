include(GoogleTest)

add_executable(
  qlearning_tests
  qlearning_tests.cpp
)
target_link_libraries(
  qlearning_tests
  GTest::gtest_main
  mcvi_library
)
gtest_discover_tests(qlearning_tests)

add_executable(
  bound_tests
  bound_tests.cpp
)
target_link_libraries(
  bound_tests
  GTest::gtest_main
  mcvi_library
)
gtest_discover_tests(bound_tests)
