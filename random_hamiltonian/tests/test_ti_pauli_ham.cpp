#include "RandomPauliHamiltonian.hpp"
#include "utils.hpp"

#include "edlib/EDP/LocalHamiltonian.hpp"
#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/Basis/TransformBasis.hpp"
#include "edlib/edlib.hpp"

#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <tbb/tbb.h>

#include <catch2/catch_all.hpp>

#include <cstdint>
#include <cstring>
#include <charconv>
#include <iostream>
#include <random>
#include <span>

TEST_CASE("Test Subspace op", "[test]") {
	std::mt19937_64 re{1337};

	const size_t N = 8;
	const size_t k = 3;

	SECTION("Random Hamiltonian") {
		for(size_t idx = 0; idx < 8; idx++) {
			const auto pauli_ham = createRandomPauliHamiltonian(re, N, k);

			edlib::Basis1D<uint32_t> basis(N, 0, false);

			const auto full_ham = edp::constructMat<std::complex<double>>(1 << N, pauli_ham);

			const auto subspace_ham = edp::constructMat<std::complex<double>>(basis.getDim(), FullBasisOpToSubspaceOp(basis, pauli_ham));

			const Eigen::MatrixXcd V = basisMatrix<uint32_t>(basis).cast<std::complex<double>>();
			const Eigen::MatrixXcd converted_mat = V.adjoint() * full_ham * V;

			REQUIRE((subspace_ham - converted_mat).squaredNorm() < 1e-5);
		}
	}

	SECTION("LocalHamiltonian") {
		edlib::Basis1D<uint32_t> basis(N, 0, false);

		edp::LocalHamiltonian<double> lh_ZZ(N, 2);
		for(int i = 0; i < N; i++) {
			lh_ZZ.addTwoSiteTerm({i, (i+1) % N}, getZZ());
		}
		const Eigen::MatrixXd V = basisMatrix<uint32_t>(basis);
		const Eigen::MatrixXd ZZ = constructMat<double>(1<<N, lh_ZZ);
		const Eigen::MatrixXd subspace_ZZ = constructMat<double>(basis.getDim(), FullBasisOpToSubspaceOp(basis, lh_ZZ));

		const Eigen::MatrixXd converted_ZZ = V.adjoint() * ZZ * V; 

		REQUIRE((subspace_ZZ - converted_ZZ).squaredNorm() < 1e-5);
	}

}

TEST_CASE("Test Ham grad in different basis") 
{
	std::mt19937_64 re{1337};
	const size_t N = 8;
	const size_t k = 3;

	const edlib::Basis1D<uint32_t> basis(N, 0, false);

	const auto pauli_ham = createRandomPauliHamiltonian(re, N, k);
	const auto full_ham = edp::constructMat<std::complex<double>>(1<<N, pauli_ham);
	const auto subspace_ham = edp::constructMat<std::complex<double>>(basis.getDim(),
			FullBasisOpToSubspaceOp(basis, pauli_ham));

	const Eigen::VectorXd full_ini = Eigen::VectorXd::Ones(1<<N) / sqrt(static_cast<double>(1 << N));
	const Eigen::VectorXd subspace_ini = [&basis, &full_ini] () -> Eigen::VectorXd {
		const auto vec = toReducedBasis(basis, std::span{full_ini.data(), size_t{1U} << N});
		return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
	}();

	edp::LocalHamiltonian<double> lh_ZZ(N, 2);
	for(int i = 0; i < N; i++) {
		lh_ZZ.addTwoSiteTerm({i, (i+1) % N}, getZZ());
	}

	double g_sqr1 = 0.0;
	double g_sqr2 = 0.0;
	{
		// full basis
		const size_t dim = size_t{1U} << N;
		auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(full_ham);

		Eigen::VectorXd evals = solver.eigenvalues();
		Eigen::MatrixXcd evecs = solver.eigenvectors();

		const Eigen::MatrixXd ZZ = constructMat<double>(1<<N, lh_ZZ);

		const Eigen::MatrixXcd G = evecs.adjoint() * ZZ * evecs;
		const auto& O = G;

		const Eigen::VectorXcd C = evecs.adjoint() * full_ini;

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

		g_sqr1 = real(2.0*sum3 - (sum1 + sum2));
	}

	{
		// reduced basis
		const size_t dim = basis.getDim();
		auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(subspace_ham);

		Eigen::VectorXd evals = solver.eigenvalues();
		Eigen::MatrixXcd evecs = solver.eigenvectors();

		const Eigen::MatrixXd ZZ = constructMat<double>(basis.getDim(), FullBasisOpToSubspaceOp(basis, lh_ZZ));

		const Eigen::MatrixXcd G = evecs.adjoint() * ZZ * evecs;
		const auto& O = G;

		const Eigen::VectorXcd C = evecs.adjoint() * subspace_ini;

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

		g_sqr2 = real(2.0*sum3 - (sum1 + sum2));
	}
	using Catch::Approx;
	REQUIRE(g_sqr1 == Approx(g_sqr2).epsilon(1e-5));
}
