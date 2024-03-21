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
  MCVI_tests
  MCVI_tests.cpp
)
target_link_libraries(
  MCVI_tests
  GTest::gtest_main
  mcvi_library
)
gtest_discover_tests(MCVI_tests)
