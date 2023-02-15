import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
import json
from utils import create_neel_st
from mpi4py import MPI

LAYERS_PER_BLOCK = 3

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

def create_circuit(num_qubits, num_blocks):
    ini_st = create_neel_st(num_qubits)

    def circuit(x):
        qml.QubitStateVector(ini_st, wires = range(num_qubits))
        for k in range(num_blocks):
            for w in range(num_qubits):
                qml.IsingXX(2*x[LAYERS_PER_BLOCK*k+0], wires=[w, (w+1) % num_qubits])
            for w in range(num_qubits):
                qml.IsingYY(2*x[LAYERS_PER_BLOCK*k+1], wires=[w, (w+1) % num_qubits])
            for w in range(num_qubits):
                qml.IsingZZ(2*x[LAYERS_PER_BLOCK*k+2], wires=[w, (w+1) % num_qubits])

        return qml.expval(qml.PauliY(0) @ qml.PauliY(1))
    
    return circuit

def sample_param(rng, *, constant, num_qubits, num_blocks):
    num_params = num_blocks * LAYERS_PER_BLOCK
    init_params = rng.random(size = num_params)

    for i in range(num_blocks):
        layer_param_sum = np.sum(init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK])
        init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK] *= (2*constant*math.pi)/(num_qubits*layer_param_sum)

    return init_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient scaling of hamiltonian variational Ansatz for the XYZ model')
    parser.add_argument('--device', required=True, type=str, choices=["lightning.qubit", "lightning.gpu", "lightning.kokkos"], help='Device')
    parser.add_argument('--num_qubits', required=True, type=int, help='Number of qubits (N)')

    args = parser.parse_args()

    device = args.device
    num_qubits = args.num_qubits
    num_blocks = 16
    num_iter = 256

    num_params = num_blocks * LAYERS_PER_BLOCK

    rng = Generator(PCG64())

    dev = qml.device(device, wires=num_qubits)
    circuit = qml.QNode(create_circuit(num_qubits, num_blocks), dev, diff_method="adjoint")

    for idx in range(mpi_rank, 101, mpi_size):
        eps = 0.01*idx
        print(f"#Processing at rank={mpi_rank}, eps={eps}", flush=True)
        grads = np.zeros((num_iter, num_blocks * LAYERS_PER_BLOCK), dtype=np.float128)

        for i in range(num_iter):
            param = pnp.array(eps*rng.random(size = num_params), requires_grad=True)
            grad = qml.grad(circuit)(param)
            grads[i,:] = grad

        filename = "grad_N{}_{:03d}.npy".format(num_qubits, int(eps*100))
        np.save(filename, grads)
