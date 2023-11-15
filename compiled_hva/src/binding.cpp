#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "hva_xyz.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_hva_xyz, m) {
	py::class_<HVA>(m, "HVA")
		.def("test", []() { return 1; });
}
