#include "utils.hpp"
#include "RandomPauliHamiltonian.hpp"

#include "edlib/EDP/LocalHamiltonian.hpp"
#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/Basis/TransformBasis.hpp"
#include "edlib/edlib.hpp"

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <tbb/tbb.h>

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <charconv>
#include <iostream>
#include <random>

uint32_t parse_uint32(char* str) {
	uint32_t value = 0;
	std::from_chars(str, str + strlen(str), value);
	return value;
}

int main(int argc, char* argv[]) {
	int mpi_size, mpi_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	if(argc != 3) {
		fmt::print("Usage {} [N] [k]\n", argv[0]);
		return 1;
	}

	fmt::print(stderr, "Processing in mpi_size = {}, mpi_rank = {}\n", mpi_size, mpi_rank);

	const uint32_t N = parse_uint32(argv[1]);
	const uint32_t k = parse_uint32(argv[2]);

	if (mpi_rank == 0) {
		nlohmann::json args_in {
			{"N", N},
			{"k", k}
		};
		std::ofstream fout("args_in.json");
		fout << args_in;
		fout.close();
	}

	std::random_device rd;
	std::mt19937_64 re{static_cast<unsigned long>(rd() + mpi_rank)};

	const edlib::Basis1D<uint32_t> basis(N, 0, false);
	const size_t dim = basis.getDim();

	const Eigen::VectorXd full_ini = Eigen::VectorXd::Ones(1<<N) / sqrt(static_cast<double>(1 << N));
	const Eigen::VectorXd subspace_ini = [&basis, &full_ini, N] () -> Eigen::VectorXd {
		const auto vec = toReducedBasis(basis, std::span{full_ini.data(), size_t{1U} << N});
		return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
	}();

	edp::LocalHamiltonian<std::complex<double>> lh_Y(N, 2);
	for(int i = 0; i < N; i++) {
		lh_Y.addOneSiteTerm(i, getY());
	}

	edp::LocalHamiltonian<double> lh_Z(N, 2);
	for(int i = 0; i < N; i++) {
		lh_Z.addOneSiteTerm(i, getZ());
	}

	const Eigen::MatrixXcd subspace_Y = constructMat<std::complex<double>>(basis.getDim(), FullBasisOpToSubspaceOp(basis, lh_Y));
	const Eigen::MatrixXd subspace_Z = constructMat<double>(basis.getDim(), FullBasisOpToSubspaceOp(basis, lh_Z));

	for(size_t ham_idx = mpi_rank; ham_idx < 1024; ham_idx += mpi_size) {
		const auto pauli_ham = createRandomPauliHamiltonian(re, N, k, true);
		const Eigen::MatrixXcd subspace_ham = edp::constructMat<std::complex<double>>(basis.getDim(),
				FullBasisOpToSubspaceOp(basis, pauli_ham));
#ifndef NDEBUG
		std::cerr << "Test infinite norm: " << subspace_ham.imag().lpNorm<Eigen::Infinity>() << std::endl;
#endif
		auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(subspace_ham.real());

		Eigen::VectorXd evals = solver.eigenvalues();
		Eigen::MatrixXd evecs = solver.eigenvectors();

		const Eigen::MatrixXcd G = evecs.adjoint() * subspace_Y * evecs;
		const Eigen::MatrixXcd O = evecs.adjoint() * subspace_Z * evecs;

		const Eigen::VectorXd C = evecs.adjoint() * subspace_ini;

		const Eigen::VectorXcd C_bar_G = C.adjoint() * G;
		const Eigen::VectorXcd GC = G * C;

		std::complex<double> sum1 = 0.0;
		for(size_t j = 0; j < dim; j++) {
			for(size_t k = 0; k < dim; k++) {
				sum1 += C_bar_G(j) * std::norm(O(j,k)) * C(k) * C_bar_G(k) *  C(j);
			}
		}

		std::complex<double> sum2 = 0.0;
		for(size_t i = 0; i < dim; i++) {
			for(size_t j = 0; j < dim; j++) {
				sum2 += std::conj(C(i)) * std::norm(O(i,j)) * GC(j) * std::conj(C(j)) * GC(i);
			}
		}

		std::complex<double> sum3 = 0.0;
		for(size_t j = 0; j < dim; j++) {
			for(size_t k = 0; k < dim; k++) {
				sum3 += C_bar_G(j)*std::norm(O(j,k)) * std::norm(C(k)) * GC(j);
			}
		}

		double g_sqr = real(2.0*sum3 - (sum1 + sum2));
		fmt::print("{}\n", g_sqr);
		fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
