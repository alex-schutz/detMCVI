if(MSVC)
add_compile_options(/W4 /WX)
else()
add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(gtest)

add_subdirectory(mcvi)
