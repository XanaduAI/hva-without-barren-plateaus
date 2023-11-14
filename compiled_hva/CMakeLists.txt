project(compiled-hva-xyz LANGUAGES CXX)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_PYTHON "Enable python binding" ON)
option(ENABLE_WARNINGS "Enable all warnings" ON)
option(BUILD_TESTS OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

add_library(hva_xyz_compile_options INTERFACE)
if(ENABLE_WARNINGS)
    target_compile_options(hva_xyz_compile_options INTERFACE -Wall -Wextra)
endif()

include(FetchContent)

set(PL_BACKEND "lightning_qubit" CACHE STRING "" FORCE)
FetchContent_Declare(
    PLLightning
    GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
    GIT_TAG v0.33.1
)
FetchContent_MakeAvailable(PLLightning)

if(ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

add_executable(test "test/test.cpp")
target_include_directories(test  PUBLIC "src")
target_link_libraries(test PUBLIC lightning_qubit hva_xyz_compile_options)
