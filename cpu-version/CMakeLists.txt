cmake_minimum_required(VERSION 3.16)
project(CS171-project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
project(CS171-final_project)

find_package(OpenMP REQUIRED)


add_subdirectory(libs)
file(GLOB SRC_FILE src/*.cpp)

add_executable(main main.cpp ${SRC_FILE})
target_link_libraries(main glm stb tinyobjloader OpenMP::OpenMP_CXX)
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)



