add_subdirectory(googletest)

include_directories(googletest/googletest/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

FILE(GLOB TEST_FILES src/*.cpp)
set(SOURCE_FILES test_all.cpp ${TEST_FILES})

add_executable(test_all ${SOURCE_FILES})
target_link_libraries(test_all curplsh gtest_main)

add_executable(test_norm test_norm.cpp)
target_link_libraries(test_norm curplsh)

add_executable(test_search test_search.cpp)
target_link_libraries(test_search curplsh)

add_test(NAME tests COMMAND test_all)
