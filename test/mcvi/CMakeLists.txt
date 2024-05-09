include(GoogleTest)


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

add_executable(
  path_tests
  path_tests.cpp
)
target_link_libraries(
  path_tests
  GTest::gtest_main
  mcvi_library
)
gtest_discover_tests(path_tests)
