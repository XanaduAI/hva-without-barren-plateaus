#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hva_xyz.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_hva_xyz, m) {
	py::class_<HVA>(m, "HVA")
		.def(py::init<size_t, size_t, const std::vector<std::pair<size_t, size_t>>&>())
		.def("expval", [](HVA& self, py::array_t<std::complex<double>,  py::array::c_style | py::array::forcecast> ini_st,
					      py::array_t<double,  py::array::c_style | py::array::forcecast> params){
			py::buffer_info buf_ini_st = ini_st.request();
			py::buffer_info buf_params = params.request();

			return self.expval(std::span{static_cast<std::complex<double>*>(buf_ini_st.ptr), static_cast<size_t>(buf_ini_st.size)},
					           std::span{static_cast<double*>(buf_params.ptr), static_cast<size_t>(buf_params.size)});
		})
		.def("expvals_shots", [](HVA& self, py::array_t<std::complex<double>,  py::array::c_style | py::array::forcecast> ini_st,
					             py::array_t<double,  py::array::c_style | py::array::forcecast> params,
				                 size_t shots) {
			py::buffer_info buf_ini_st = ini_st.request();
			py::buffer_info buf_params = params.request();

			return self.expval_shots(
					std::span{static_cast<std::complex<double>*>(buf_ini_st.ptr), static_cast<size_t>(buf_ini_st.size)},
					std::span{static_cast<double*>(buf_params.ptr), static_cast<size_t>(buf_params.size)},
					shots
			);
		})
		.def("grad_shots", [](HVA& self, py::array_t<std::complex<double>,  py::array::c_style | py::array::forcecast> ini_st,
					          py::array_t<double,  py::array::c_style | py::array::forcecast> params,
				              size_t shots){
			py::buffer_info buf_ini_st = ini_st.request();
			py::buffer_info buf_params = params.request();

			return self.grad_shots(
					std::span{static_cast<std::complex<double>*>(buf_ini_st.ptr), static_cast<size_t>(buf_ini_st.size)},
					std::span{static_cast<double*>(buf_params.ptr), static_cast<size_t>(buf_params.size)},
					shots
			);
		});
}
