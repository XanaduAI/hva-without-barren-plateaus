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
	const uint32_t N = parse_uint32(argv[1]);
	const uint32_t k = parse_uint32(argv[2]);

	std::mt19937_64 re{1337 };

	for(size_t ham_idx = 0; ham_idx < 32; ham_idx ++) {
		const auto pauli_ham = createRandomPauliHamiltonian(re, N, k, true);

		std::cerr << pauli_ham.params() << std::endl;
	}

	return 0;
}
