#pragma once
#include "edlib/EDP/LocalHamiltonian.hpp"
#include "edlib/EDP/ConstructSparseMat.hpp"

#include <Eigen/Sparse>

#include <charconv>

Eigen::SparseMatrix<double> getZZ();
Eigen::SparseMatrix<double> getZ();
Eigen::SparseMatrix<double> getX();
Eigen::SparseMatrix<std::complex<double>> getY();

inline Eigen::SparseMatrix<double> getG(uint32_t N) {
	edp::LocalHamiltonian<double> lh(N, 2);
	lh.addTwoSiteTerm({0, 1}, getZZ());
	//lh.addOneSiteTerm(0, getX());
	return edp::constructSparseMat<double>(1U << N, lh);
}

inline Eigen::SparseMatrix<double> getO(uint32_t N, uint32_t i) {
	edp::LocalHamiltonian<double> lh(N, 2);
	lh.addTwoSiteTerm({i, (i+1)%N}, getZZ());
	return edp::constructSparseMat<double>(1U << N, lh);
}

uint32_t get_num_threads();

inline Eigen::MatrixXd non_integrable_ham(const uint32_t N) {
	constexpr static double g = 0.9045;
	constexpr static double h = 0.8090;

	edp::LocalHamiltonian<double> lh(N, 2);

	for(uint32_t i = 0; i < N; i++) {
		lh.addTwoSiteTerm({i, (i+1) % N}, getZZ());
		lh.addOneSiteTerm(i, h*getZ());
		lh.addOneSiteTerm(i, g*getX());
	}

	return edp::constructMat<double>(1U << N, lh);
}

template <typename Basis, typename FullBasisOp>
class FullBasisOpToSubspaceOp {
private:
	const Basis& basis_;
	const FullBasisOp& full_op_;

public:
	FullBasisOpToSubspaceOp(const Basis& basis, const FullBasisOp& full_op)
		: basis_{basis}, full_op_{full_op} {}

	auto operator()(uint32_t col) const {
		const auto rpt = basis_.getNthRep(col);
		auto m = full_op_(rpt);
		using T = typename decltype(m)::mapped_type;

		std::map<std::size_t, T> res;

		for(const auto& [s, coeff]: m) {
			const auto [bidx, ham_coeff] = basis_.hamiltonianCoeff(s, col);
			if(bidx >= 0)
			{
				res[bidx] += ham_coeff * coeff;
			}
		}
		return res;
	}
};

template<typename UINT, template<typename> class Basis>
Eigen::MatrixXd basisMatrix(const Basis<UINT>& basis)
{
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(1U << basis.getN(), basis.getDim());
    for(unsigned int n = 0; n < basis.getDim(); ++n)
    {
        auto bvec = basis.basisVec(n);
        for(const auto& p : bvec)
        {
            res(p.first, n) = p.second;
        }
    }
    return res;
}
