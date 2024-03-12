add_executable(
  qlearning_tests
  qlearning_tests.cpp
)

target_link_libraries(
  qlearning_tests
  GTest::gtest_main
  mcvi_library
)

include(GoogleTest)
gtest_discover_tests(qlearning_tests)
