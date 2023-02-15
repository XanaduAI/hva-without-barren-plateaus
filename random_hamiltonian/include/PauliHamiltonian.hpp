#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

enum class Pauli : char
{
	I = 'I',
	X = 'X',
	Y = 'Y',
	Z = 'Z'
};

inline
std::strong_ordering operator<=>(const Pauli& lhs, const Pauli& rhs) {
	return static_cast<char>(lhs) <=> static_cast<char>(rhs);
}

inline
std::optional<Pauli> charToPauli(char c)
{
	switch(c)
	{
	case 'I':
		return Pauli::I;
	case 'X':
		return Pauli::X;
	case 'Y':
		return Pauli::Y;
	case 'Z':
		return Pauli::Z;
	}
	return {};
}

class PauliString
{
private:
	// sorted
	std::vector<std::pair<uint32_t, Pauli>> string_;

public:
	PauliString() { }

	PauliString(std::string str)
	{
		if(str.empty())
			return;
		while(true)
		{
			std::size_t pos = str.find(' ');
			size_t where;
			std::string token = str.substr(0, pos);
			where = atoi(token.c_str() + 1);
			auto p = charToPauli(token[0]);

			if(!p.has_value()) {
				throw std::invalid_argument("Pauli operator cannot be parsed");
			}
			string_.emplace_back(where, p.value());
			if(pos != std::string::npos)
				str.erase(0, pos + 1);
			else
				break;
		}
		std::sort(string_.begin(), string_.end());
	}

	bool operator==(const PauliString& rhs) const {
		if(string_.size() != rhs.string_.size()) {
			return false;
		}

		for(size_t i = 0; i < string_.size(); i++) {
			if(string_[i] != rhs.string_[i]) {
				return false;
			}
		}
		return true;
	}

	uint32_t countX() const
	{
		uint32_t res = 0;
		for(const auto& p : string_)
		{
			if(p.second == Pauli::X)
			{
				++res;
			}
		}
		return res;
	}

	uint32_t countY() const
	{
		uint32_t res = 0;
		for(const auto& p : string_)
		{
			if(p.second == Pauli::Y)
			{
				++res;
			}
		}
		return res;
	}

	uint32_t countZ() const
	{
		uint32_t res = 0;
		for(const auto& p : string_)
		{
			if(p.second == Pauli::Z)
			{
				++res;
			}
		}
		return res;
	}

	/**
	 * Check whether the term is diagonal in the computational basis
	 */
	bool isDiagonal() const
	{
		for(const auto& p : string_)
		{
			if((p.second != Pauli::Z) && (p.second != Pauli::I))
			{
				return false;
			}
		}
		return true;
	}

	bool isEmpty() const {
		return string_.empty();
	}

	bool hasU1() const { return (countX() % 2 == 0) && (countY() % 2 == 0); }

	void add(uint32_t pos, Pauli p) {
		auto elt = std::make_pair(pos, p);
		const auto it = std::lower_bound(string_.begin(), string_.end(), elt);
		if((it != string_.end()) && it->first == pos) {
			throw std::invalid_argument("Cannot add to an existing position");
		}
		string_.emplace(it, std::move(elt));
	}

	void simplify() {
		auto iter = string_.begin();
		while(iter != string_.end()) {
			if(iter->second == Pauli::I) {
				iter = string_.erase(iter);
			} else {
				++iter;
			}
		}
	}
	
	template<typename T>
	std::pair<std::complex<T>, uint32_t> apply(uint32_t col) const
	{
		constexpr std::complex<int> I(0, 1);
		std::complex<int> r = 1;
		std::vector<int> toFlip;
		for(const auto p : string_)
		{
			int sgn = 1 - 2 * ((col >> p.first) & 1);
			switch(p.second)
			{
			case Pauli::I:
				break;
			case Pauli::X:
				col ^= (1 << p.first);
				break;
			case Pauli::Y:
				col ^= (1 << p.first);
				r *= sgn * I;
				break;
			case Pauli::Z:
				r *= sgn;
				break;
			}
		}
		return std::make_pair(std::complex<T>(real(r), imag(r)), col);
	}

	friend std::ostream& operator<<(std::ostream& os, const PauliString& ps)
	{
		const char* sep = "";
		for(const auto& p : ps.string_)
		{
			os << sep << static_cast<char>(p.second) << p.first;
			sep = " ";
		}
		return os;
	}
};

template<typename T>
class PauliHamiltonian
{
private:
	uint32_t n_;
	uint32_t nup_; // only effective when the Hamiltonain has U1 symmetry.
	std::vector<std::pair<T, PauliString>> terms_;

public:
	PauliHamiltonian(uint32_t n, uint32_t jz = std::numeric_limits<uint32_t>::max())
		: n_{n}, nup_{jz}
	{
	}

	uint32_t getN() const { return n_; }

	PauliHamiltonian(const PauliHamiltonian&) = default;
	PauliHamiltonian(PauliHamiltonian&&) = default;

	PauliHamiltonian& operator=(const PauliHamiltonian&) = default;
	PauliHamiltonian& operator=(PauliHamiltonian&&) = default;

	nlohmann::json params() const
	{
		using nlohmann::json;
		json res;
		res["name"] = "PauliHamiltonian";
		res["n"] = n_;
		json terms = json::array();
		;
		for(auto [coeff, pauliString] : terms_)
		{
			std::ostringstream ss;
			ss << pauliString;
			std::string s = ss.str();
			terms.push_back({coeff, std::move(s)});
		}
		res["terms"] = terms;
		return res;
	}

	size_t numTerms() const {
		return terms_.size();
	}

	template<typename... Ts> void emplaceTerm(Ts&&... args)
	{
		terms_.emplace_back(std::forward<Ts>(args)...);
	}

	bool hasTermwiseU1() const
	{
		return std::all_of(terms_.begin(), terms_.end(),
		                   [](const std::pair<T, PauliString>& p)
		                   { return p.second.hasU1(); });
	}

	bool isReal() const
	{
		return std::all_of(terms_.begin(), terms_.end(),
		                   [](const std::pair<T, PauliString>& p)
		                   { return (p.second.countY() % 2) == 0; });
	}

	uint32_t getNup() const { return nup_; }

	static PauliHamiltonian fromFile(const std::filesystem::path& filePath)
	{
		std::ifstream fin(filePath);
		std::string line;
		std::getline(fin, line);
		uint32_t n;
		uint32_t m;
		std::istringstream ss(line);
		ss >> n >> m;

		PauliHamiltonian res(n, m);
		while(std::getline(fin, line))
		{
			auto to = line.find("\t");
			T coeff = stof(line.substr(0, to));
			PauliString s(line.substr(to + 1, std::string::npos));
			res.emplaceTerm(coeff, s);
		}
		return res;
	}

	template<typename State> typename State::Scalar operator()(const State& state) const
	{
		typename State::Scalar res = 0;
		for(auto [coeff, pauliString] : terms_)
		{
			const auto& [c, toFlip] = pauliString(state.getSigma());
			res += c * coeff * state.ratio(toFlip);
		}
		return res;
	}

	void simplify() {
		for(auto& elt: terms_) {
			elt.second.simplify();
		}
	}

	std::map<uint32_t, std::complex<T>> operator()(const uint32_t col) const
	{
		std::map<uint32_t, std::complex<T>> m;
		for(auto [coeff, pauliString] : terms_)
		{
			const auto& [c, s] = pauliString.template apply<T>(col);
			m[s] += c * coeff;
		}
		return m;
	}

	friend std::ostream& operator<<(std::ostream& os, const PauliHamiltonian& ham)
	{
		const char* sep = "";
		for(const auto& [coeff, p_str]: ham.terms_)
		{
			os << '[' << coeff << ", " << p_str << ']';
			sep = ", ";
		}
		return os;
	}

	friend PauliHamiltonian diagonalHamiltonian(const PauliHamiltonian& ph)
	{
		PauliHamiltonian res(ph.n_, ph.nup_);
		for(auto [coeff, pauliString] : ph.terms_)
		{
			if(pauliString.isDiagonal())
				res.emplaceTerm(coeff, pauliString);
		}
		return res;
	}
};
