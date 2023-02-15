#pragma once
#include "PauliHamiltonian.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>
#include <set>

#include <sstream>

std::vector<size_t> generatePositions(const size_t N, const size_t k, const size_t starting_idx) {
	std::vector<size_t> positions(N, 0);
	std::iota(positions.begin(), positions.end(), starting_idx);
	for(auto& p: positions) {
		p %= N;
	}
	return positions;
}

PauliString generatePauliTermStr(const std::string& str, const std::vector<size_t>& positions) {
	PauliString pstr;

	for(size_t i = 0; i < str.size(); i++) {
		pstr.add(positions[i], Pauli{str[i]});
	}
	return pstr;
}

constexpr size_t ipow(size_t base, size_t exponent) {
	size_t res = 1;
	for(size_t i = 0; i < exponent; i++) {
		res *= base;
	}
	return res;
}

namespace {
std::set<std::string> pauliTermsUniqueTranslation(size_t k) {
	std::array<char, 4> paulis{'I', 'X', 'Y', 'Z'};
	std::set<std::string> pauli_strs;

	for(size_t i = 1; i < ipow(4, k); i++) {
		std::vector<char> term;
		size_t m = i;
		for(size_t n = 0; n < k; n++) {
			term.emplace_back(paulis[m % 4]);
			m /= 4;
		}

		const auto it_b = std::find_if(term.begin(), term.end(), [](char c) {
			return c != 'I';
		});
		term.erase(term.begin(), it_b);

		const auto it_e = std::find_if(term.rbegin(), term.rend(), [](char c) {
			return c != 'I';
		});
		term.erase(it_e.base(), term.end());
		pauli_strs.emplace(term.begin(), term.end());
	}
	return pauli_strs;
}
}

size_t countY(const std::string& str) {
	size_t res = 0;
	for(const auto& c: str) {
		if (c == 'Y') {
			res ++;
		}
	}
	return res;
}

template <typename RandomEngine>
PauliHamiltonian<double> createRandomPauliHamiltonian(RandomEngine& re, const size_t N, const size_t k, bool only_real = false) {
	assert(k >= 1);
	std::normal_distribution<double> ndist;
	auto pauli_strs = pauliTermsUniqueTranslation(k);

	if (only_real) {
		auto iter = std::cbegin(pauli_strs);
		while(iter != std::cend(pauli_strs)) {
			if ((countY(*iter) % 2) != 0) {
				// odd number of Y
				iter = pauli_strs.erase(iter);
			} else {
				++iter;
			}
		}
	}

	PauliHamiltonian<double> ham(N);
	for(const auto& pauli_str: pauli_strs) {
		const double coeff = ndist(re);
		for(size_t n = 0; n < N; n++) {
			auto positions = generatePositions(N, k, n);
			auto term_str = generatePauliTermStr(pauli_str, positions);
			ham.emplaceTerm(coeff, term_str);
		}
	}
	ham.simplify();
	return ham;
}
