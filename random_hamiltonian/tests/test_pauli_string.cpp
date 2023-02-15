#include "PauliHamiltonian.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Test PauliString", "[PauliString]") {
	REQUIRE(PauliString("X0 Y3") == PauliString("Y3 X0"));

	SECTION("Add") {
		auto p1 = PauliString("Y3 X0");
		p1.add(1, Pauli('Z'));
		p1.simplify();
		REQUIRE(p1 == PauliString("X0 Y3 Z1"));
		REQUIRE(p1 == PauliString("X0 Z1 Y3"));
	}
	SECTION("Add and simplify") {
		auto p1 = PauliString("X0 Y3");
		p1.add(1, Pauli('I'));
		p1.simplify();
		REQUIRE(p1 == PauliString("X0 Y3"));
	}
}
