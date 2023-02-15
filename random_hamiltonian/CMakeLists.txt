project(RandomHamiltonianGrad LANGUAGES CXX)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_TESTS OFF)

################################################################################
#                              Set dependencies
################################################################################

include(FetchContent)
FetchContent_Declare(
    ExactDiagonalization
    GIT_REPOSITORY https://github.com/chaeyeunpark/ExactDiagonalization.git
    GIT_TAG master
)
FetchContent_MakeAvailable(ExactDiagonalization)

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        9.0.0)
FetchContent_GetProperties(fmt)
if (NOT fmt_POPULATED)
    FetchContent_Populate(fmt)
    add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
find_package(arpack-ng REQUIRED)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

add_library(ham_grad_dependency INTERFACE)
target_include_directories(ham_grad_dependency INTERFACE externals include)
target_sources(ham_grad_dependency INTERFACE src/utils.cpp)

add_executable(random_ham_grad src/random_ham_grad.cpp)
target_link_libraries(random_ham_grad EDLIB::edlib ham_grad_dependency MPI::MPI_CXX fmt::fmt ARPACK::ARPACK OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

add_executable(random_ham_grad_real src/random_ham_grad_real.cpp)
target_link_libraries(random_ham_grad_real EDLIB::edlib ham_grad_dependency MPI::MPI_CXX fmt::fmt ARPACK::ARPACK OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

add_executable(print_generated_random_ham src/print_generated_random_ham.cpp)
target_link_libraries(print_generated_random_ham EDLIB::edlib ham_grad_dependency fmt::fmt)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
