#include <algorithm>
#include <cstdlib>
#include <random>
#include <span>
#include <stack>
#include <unordered_map>
#include <vector>

#include <omp.h>

class RandomEngineManager {
private:
	std::vector<std::mt19937_64> random_engines_;
	RandomEngineManager() {
		std::random_device rd;

		for(size_t i = 0; i < static_cast<size_t>(omp_get_max_threads()); i++) {
			random_engines_.emplace_back(rd());
		}
	}

public:
	std::mt19937_64& get_random_engine() {
		return random_engines_[omp_get_thread_num()];
	}

	static RandomEngineManager& get_instance() {
		static RandomEngineManager s;
		return s;
	}
};

std::vector<size_t> generate_samples_alias(size_t num_samples, const std::span<const double> probabilities) {
	size_t N = probabilities.size();


	std::vector<double> prob_table(N, 0.0);
	std::vector<size_t> alias_table(N, 0);

	{
		// Table generation
		std::stack<size_t> overfull_indcs;
		std::stack<size_t> underfull_indcs;

		for(size_t idx = 0; idx < N; idx++) {
			double val = N * probabilities[idx];
			prob_table[idx] = val;
			alias_table[idx] = idx;

			if(val > 1.0) {
				overfull_indcs.push(idx);
			} else {
				underfull_indcs.push(idx);
			}
		}

		while((!overfull_indcs.empty()) && (!underfull_indcs.empty())) {
			const size_t i = overfull_indcs.top();
			const size_t j = underfull_indcs.top();
			underfull_indcs.pop();

			alias_table[j] = i;
			prob_table[i] += prob_table[j] - 1;
			// now j is in full category

			if(prob_table[i] < 1.0) {
				overfull_indcs.pop();
				underfull_indcs.push(i);
			}
		}
	}

	std::vector<size_t> samples(num_samples, 0UL);
	// Pick samples
#pragma omp parallel shared(samples, prob_table, alias_table)
	{
		std::uniform_real_distribution<double> distribution(0.0, 1.0);
		std::mt19937_64& re = RandomEngineManager::get_instance().get_random_engine();

#pragma omp for schedule(static, 8)
		for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
			const double p = distribution(re);
			size_t i = static_cast<size_t>(N*p);
			double y = N*p - i; // Uniformly distributed in [0,1)

			if(y < prob_table[i]) {
				samples[sample_idx] = i;
			} else {
				samples[sample_idx] = alias_table[i];
			}
		}
	}

	return samples;
}

std::vector<size_t> generate_samples(size_t num_samples, const std::span<const double> probabilities) {
	std::vector<double> accum(probabilities.size(), 0.0);
	accum[0] = probabilities[0];
	for(size_t i = 1; i < probabilities.size(); i++) {
		accum[i] = accum[i-1] + probabilities[i];
	}


	std::vector<size_t> samples(num_samples, 0UL);
	// Pick samples
#pragma omp parallel shared(samples, accum)
	{
		std::uniform_real_distribution<double> distribution(0.0, 1.0);
		std::mt19937_64& re = RandomEngineManager::get_instance().get_random_engine();

#pragma omp for schedule(static, 8)
		for (size_t i = 0; i < num_samples; i++) {
			double p = distribution(re);
			const auto iter = std::upper_bound(accum.begin(), accum.end(), p);
			samples[i] = std::distance(accum.begin(), iter);
		}
	}

	return samples;
}
