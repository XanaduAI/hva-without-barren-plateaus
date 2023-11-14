#pragma once
#include "sample.hpp"

#include "StateVectorLQubitManaged.hpp"

#include <algorithm>
#include <complex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <numbers>

template<typename T>
void fwht_inplace(std::span<T> vec)
{
	std::vector<T> transformed(vec.size());

	size_t h = 1;
	while(h < vec.size())
	{
#pragma omp parallel for shared(vec,transformed) firstprivate(h)
		for(size_t i = 0; i < vec.size(); i += 2*h) {
			std::transform(vec.begin() + i, vec.begin() + i + h,
					vec.begin() + i + h, transformed.begin() + i,
					std::plus<T>());
			std::transform(vec.begin() + i , vec.begin() + i + h,
					vec.begin() + i + h, transformed.begin() + i + h,
					std::minus<T>());
		}
		std::copy(transformed.begin(), transformed.end(), vec.begin());
		h *= 2;
	}
}

class HVA {
private:
	size_t num_qubits_;
	size_t num_blocks_;
	std::vector<std::pair<size_t, size_t>> edges_;

	double expval_shots_allparams(std::span<const std::complex<double>> ini_st, const std::span<const double> params, size_t shots) const {
		if(params.size() != 3 * num_blocks_ * edges_.size()) {
			throw std::invalid_argument("Size of the parameters mismatch.");
		}
		Pennylane::LightningQubit::StateVectorLQubitManaged<double> sv{num_qubits_};
		sv.updateData(ini_st.data(), ini_st.size());

		size_t p_idx = 0;

		for(size_t k = 0; k < num_blocks_; k++) {
			for(const auto& edge: edges_) {
				sv.applyOperation("IsingXX", {edge.first, edge.second}, false, {params[p_idx++]});
			}
			for(const auto& edge: edges_) {
				sv.applyOperation("IsingYY", {edge.first, edge.second}, false, {params[p_idx++]});
			}
			for(const auto& edge: edges_) {
				sv.applyOperation("IsingZZ", {edge.first, edge.second}, false, {params[p_idx++]});
			}
		}

		const double dim = 1UL << num_qubits_;
	
		auto st = sv.getDataVector(); // state in Z basis

		double (*norm)(const std::complex<double>& ) = std::norm<double>;
		std::vector<double> probs(st.size(), 0.0);
		std::transform(st.begin(), st.end(), probs.begin(), norm);

		// Compute zz obs
		const auto samples_z = generate_samples_alias(shots, probs);

		double zz = 0.0;
#pragma omp parallel shared(samples_z,edges_) reduction(+:zz)
		{
			double zz_loc = 0;
#pragma omp for
			for(size_t sample_idx = 0; sample_idx < shots; sample_idx++) {
				const auto sample = samples_z[sample_idx];
				for(auto&& [i,j]: edges_) {
					const size_t rev_i = num_qubits_ - i - 1;
					const size_t rev_j = num_qubits_ - j - 1;
					zz_loc += (1-2*int((sample >> rev_i) & 1U))*(1-2*int((sample >> rev_j) & 1U));
				}
			}
			zz += zz_loc;
		}
		zz /= shots;
		
		// Compute xx obs
		fwht_inplace(std::span{st.begin(), st.end()});

#pragma omp parallel for
		for(size_t i = 0; i < st.size(); i++) {
			st[i] /= std::sqrt(dim);
		}
		std::transform(st.begin(), st.end(), probs.begin(), norm);

		const auto samples_x = generate_samples_alias(shots, probs);

		double xx = 0.0;
#pragma omp parallel shared(samples_z)  reduction(+:xx)
		{
			double xx_loc = 0;
#pragma omp for
			for(size_t sample_idx = 0; sample_idx < shots; sample_idx++) {
				const auto sample = samples_x[sample_idx];
				for(auto&& [i,j]: edges_) {
					const size_t rev_i = num_qubits_ - i - 1;
					const size_t rev_j = num_qubits_ - j - 1;
					xx_loc += (1-2*int((sample >> rev_i) & 1U))*(1-2*int((sample >> rev_j) & 1U));
				}
			}
			xx += xx_loc;
		}

		xx /= shots;

		// Compute yy obs
		st = sv.getDataVector();
#pragma omp parallel for
		for(size_t i = 0; i < st.size(); i++) {
			st[i] *= std::pow(std::complex<double>{0.0, -1.0}, std::popcount(i));
		}
		fwht_inplace(std::span{st.begin(), st.end()});
#pragma omp parallel for
		for(size_t i = 0; i < st.size(); i++) {
			st[i] /= std::sqrt(dim);
		}
		std::transform(st.begin(), st.end(), probs.begin(), norm);

		const auto samples_y = generate_samples_alias(shots, probs);

		double yy = 0.0;
#pragma omp parallel shared(samples_y) reduction(+:yy)
		{
			double yy_loc = 0;
#pragma omp for
			for(size_t sample_idx = 0; sample_idx < shots; sample_idx++) {
				const auto sample = samples_y[sample_idx];
				for(auto&& [i,j]: edges_) {
					const size_t rev_i = num_qubits_ - i - 1;
					const size_t rev_j = num_qubits_ - j - 1;
					yy_loc += (1-2*int((sample >> rev_i) & 1U))*(1-2*int((sample >> rev_j) & 1U));
				}
			}
			yy += yy_loc;
		}

		yy /= shots;

		return xx + yy + zz;
	}
public:
	HVA(size_t num_qubits, size_t num_blocks, const std::span<const std::pair<size_t, size_t>> edges)
		: num_qubits_{num_qubits}, num_blocks_{num_blocks}, edges_(edges.begin(), edges.end()){
	}


	double expval_shots(std::span<const std::complex<double>> ini_st, const std::span<const double> params, size_t shots) const {
		if(params.size() != 3 * num_blocks_) {
			throw std::invalid_argument("Size of the parameters mismatch.");
		}

		std::vector<double> all_params;
		for(const auto p: params) {
			for(size_t i = 0; i < edges_.size(); i++) {
				all_params.push_back(p);
			}
		}

		return expval_shots_allparams(ini_st, all_params, shots);
	}

	std::vector<double> grad_shots(std::span<const std::complex<double>> ini_st, const std::span<const double> params, size_t shots) const {
		std::vector<double> all_params;
		for(const auto p: params) {
			for(size_t i = 0; i < edges_.size(); i++) {
				all_params.push_back(p);
			}
		}

		std::vector<double> all_grads;
		for(size_t i = 0; i < all_params.size(); i++) {
			const auto val_prev = all_params[i];
			all_params[i] = val_prev + std::numbers::pi_v<double>/2;
			double e1 = expval_shots_allparams(ini_st, all_params, shots);
			all_params[i] = val_prev - std::numbers::pi_v<double>/2;
			double e2 = expval_shots_allparams(ini_st, all_params, shots);

			all_params[i] = val_prev;

			all_grads.push_back((e1 - e2) / 2);
		}

		std::vector<double> grads(params.size(), 0.0);
		for(size_t i = 0; i < params.size(); i++) {
			size_t starting_idx = edges_.size() * i;
			for(size_t e_idx = 0; e_idx < edges_.size(); e_idx++) {
				grads[i] += all_grads[starting_idx + e_idx];
			}
		}

		return grads;
	}
};
