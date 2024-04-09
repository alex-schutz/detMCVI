add_executable(tiger_experiment tiger.cpp)
target_link_libraries(tiger_experiment mcvi_library)

add_executable(ctp_experiment CTP.cpp)
target_link_libraries(ctp_experiment mcvi_library)
