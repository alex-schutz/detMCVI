cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
project(MCVI)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

if ( MSVC )
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
add_subdirectory(experiments)

