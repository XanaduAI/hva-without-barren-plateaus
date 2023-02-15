## C++ code for computing the lower bound of gradients for random many-body Hamiltonian

This directory contains source code for generating data for Figure 2 of the paper.

Compilers with proper C++20 support, CMake, OpenMP, Eigen3, MPI, and ARPACK-NG are required to install this package. For example, in Ubuntu 22.04, 

```bash
$ sudo apt install g++ openmpi cmake
```

will install suitable version of GCC, OpenMPI (an implementation of MPI), and CMake. Eigen and ARPACK-NG can be installed following instructions in [eigen homepage](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [ARPACK-NG](https://github.com/opencollab/arpack-ng) repository, respectively.


Then one can build the code as follows:

```bash
$ cmake . -Bbuild 
$ cmake --build ./build
```

After that the resulting executable files `random_ham_grad` and `random_ham_grad_real` will be available within `build` directory.

Feel free to open an issue when having trouble in compiling the package.
