add_executable(maze_experiment maze.cpp)
target_link_libraries(maze_experiment mcvi_library)

add_executable(maze_timeseries maze_time_series.cpp)
target_link_libraries(maze_timeseries mcvi_library)
