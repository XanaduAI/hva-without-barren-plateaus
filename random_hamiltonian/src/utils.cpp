#include "utils.hpp"

Eigen::SparseMatrix<double> getZZ() {
	Eigen::SparseMatrix<double> m(4,4);
	m.coeffRef(0, 0) = 1.0;
	m.coeffRef(1, 1) = -1.0;
	m.coeffRef(2, 2) = -1.0;
	m.coeffRef(3, 3) = 1.0;

	m.makeCompressed();
	return m;
}

Eigen::SparseMatrix<double> getZ() {
	Eigen::SparseMatrix<double> m(2,2);
	m.coeffRef(0, 0) = 1.0;
	m.coeffRef(1, 1) = -1.0;
	m.makeCompressed();
	return m;
}

Eigen::SparseMatrix<std::complex<double>> getY() {
	Eigen::SparseMatrix<std::complex<double>> m(2,2);
	m.coeffRef(0, 1) = -std::complex{0.0, 1.0};
	m.coeffRef(1, 0) = std::complex{0.0, 1.0};
	m.makeCompressed();
	return m;
}


Eigen::SparseMatrix<double> getX() {
	Eigen::SparseMatrix<double> m(2,2);
	m.coeffRef(0, 1) = 1.0;
	m.coeffRef(1, 0) = 1.0;
	m.makeCompressed();
	return m;
}

uint32_t get_num_threads() {
	const char* str = getenv("TBB_NUM_THREADS");
	uint32_t num_threads;
	std::from_chars(str, str + strlen(str), num_threads);
	return num_threads;
}
