project(RandomHamiltonianGrad-Tests LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.1.0
)

FetchContent_MakeAvailable(Catch2)
get_target_property(Catch2_INC_DIR Catch2 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(Catch2 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Catch2_INC_DIR}")
# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

add_executable(runner test_ti_pauli_ham.cpp
                      test_pauli_string.cpp)
target_link_libraries(runner PRIVATE EDLIB::edlib Eigen3::Eigen ham_grad_dependency Catch2::Catch2WithMain)

catch_discover_tests(runner)
