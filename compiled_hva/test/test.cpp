#include "hva_xyz.hpp"
#include <iostream>
#include <numbers>

int main() {
	std::vector<std::pair<size_t, size_t>> edges;
	for(size_t i = 0; i < 16; i++) {
		edges.emplace_back(i, (i+1)%16);
	}
	edges.emplace_back(1, 3);

	HVA hva(16, 3, std::span{edges});

	std::vector<double> params(9, 0.2);

	params[3] = 0.8;

	std::vector<std::complex<double>> ini_st(1u << 16, 0.0);
	ini_st[0] = 1.0;

	std::cout << hva.expval(std::span{ini_st}, std::span{params}) << std::endl;
	std::cout << hva.expval_shots(std::span{ini_st}, std::span{params}, 1'000'000) << std::endl;

	auto grads = hva.grad_shots(std::span{ini_st}, std::span{params}, 1'000'000);
	for(const auto g: grads) {
		std::cout << g << ", ";
	}
	std::cout << std::endl;

	return 0;
}
