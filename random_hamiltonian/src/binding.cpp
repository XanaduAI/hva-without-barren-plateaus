#include "PauliHamiltonian.hpp"
#include "utils.hpp"

#include "edlib/Basis/Basis1D.hpp"
#include "edlib/EDP/ConstructSparseMat.hpp"
#include "edlib/Basis/TransformBasis.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace edlib;

Eigen::SparseMatrix<std::complex<double>> pauli_str_to_ham(uint64_t num_wires, const PauliString& p_str) {
    const uint64_t dim = uint64_t{1U} << num_wires;
    
    Eigen::SparseMatrix<std::complex<double>> m(dim, dim);

    for(uint64_t col = 0; col < dim; ++col) {
        const auto p = p_str.apply<double>(col);
        m.coeffRef(p.second, col) = p.first;
    }
    return m;
}

PYBIND11_MODULE(pauli_hamiltonian_builder, m) {
    py::enum_<Pauli>(m, "Pauli")
        .value("I", Pauli::I)
        .value("X", Pauli::X)
        .value("Y", Pauli::Y)
        .value("Z", Pauli::Z)
        .export_values();

    py::class_<PauliString>(m, "PauliString")
        .def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("add", &PauliString::add)
        .def("simplify", &PauliString::simplify)
		.def("__repr__", [](const PauliString& ps) -> std::string {
			std::ostringstream ss;
			ss << ps;
			return ss.str();
		});

    py::class_<PauliHamiltonian<double>>(m, "PauliHamiltonian")
        .def(py::init<const uint64_t>())
        .def("add_term", [](PauliHamiltonian<double>& self, double coeff, const PauliString& str) {
            self.emplaceTerm(coeff, str);
        })
        .def("to_sparse_mat", [](PauliHamiltonian<double>& self) {
            return edp::constructSparseMat<std::complex<double>>(1 << self.getN(), self);
        })
		.def("__repr__", [](const PauliHamiltonian<double>& pham) -> std::string {
			std::ostringstream ss;
			ss << pham;
			return ss.str();
		})
		.def("num_terms", &PauliHamiltonian<double>::numTerms);

	py::class_<Basis1D<uint32_t>>(m, "Basis1D")
		.def(py::init<uint32_t, uint32_t, bool>())
		.def("get_dim", &Basis1D<uint32_t>::getDim)
		.def("get_num_wires", &Basis1D<uint32_t>::getN)
		.def("basis_vec", &Basis1D<uint32_t>::basisVec);

	using NumpyArrC = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
	using NumpyArrR = py::array_t<double, py::array::c_style | py::array::forcecast>;

	m.def("pauli_ham_to_subspace_ham",
		[](const Basis1D<uint32_t>& basis, const PauliHamiltonian<double>& pauli_ham)  {
			const auto subspace_ham = FullBasisOpToSubspaceOp(basis, pauli_ham);
			return edp::constructSparseMat<std::complex<double>>(basis.getDim(), subspace_ham);
		}
	);

	m.def("to_reduced_basis",
		[](const Basis1D<uint32_t>& basis, const NumpyArrR& vec) {
		const auto buffer = vec.request();
		const auto num_wires = basis.getN();
		if (buffer.size != (size_t{1U} << num_wires)) {
			throw std::invalid_argument("Size of the input vector must be 2**N.");
		}
		return toReducedBasis(basis, {static_cast<double*>(buffer.ptr), static_cast<size_t>(buffer.size)});
	});
	m.def("to_original_basis",
		[](const Basis1D<uint32_t>& basis, NumpyArrR& vec) {
		const auto buffer = vec.request();
		if (buffer.size != basis.getDim()) {
			throw std::invalid_argument("Size of the input vector must be the same as the subspace dimension.");
		}
		return toOriginalBasis(basis, {static_cast<double*>(buffer.ptr), static_cast<size_t>(buffer.size)});
	});
}
