import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
import json
from utils import create_neel_st_2d

LAYERS_PER_BLOCK = 3

def create_circuit(Lx, Ly, num_blocks):
    ini_st = create_neel_st_2d(Lx, Ly)
    def wire_idx(i, j):
        return j*Lx + i

    def circuit(x):
        qml.QubitStateVector(ini_st, wires = range(num_qubits))
        for k in range(num_blocks):
            for i in range(Lx):
                for j in range(Ly):
                    curr_wire = wire_idx(i, j)
                    qml.IsingXX(2*x[LAYERS_PER_BLOCK*k+0], wires=[curr_wire, wire_idx((i+1)%Lx, j)])
                    qml.IsingXX(2*x[LAYERS_PER_BLOCK*k+0], wires=[curr_wire, wire_idx(i, (j+1)%Ly)])
            for i in range(Lx):
                for j in range(Ly):
                    curr_wire = wire_idx(i, j)
                    qml.IsingYY(2*x[LAYERS_PER_BLOCK*k+1], wires=[curr_wire, wire_idx((i+1)%Lx, j)])
                    qml.IsingYY(2*x[LAYERS_PER_BLOCK*k+1], wires=[curr_wire, wire_idx(i, (j+1)%Ly)])
            for i in range(Lx):
                for j in range(Ly):
                    curr_wire = wire_idx(i, j)
                    qml.IsingZZ(2*x[LAYERS_PER_BLOCK*k+2], wires=[curr_wire, wire_idx((i+1)%Lx, j)])
                    qml.IsingZZ(2*x[LAYERS_PER_BLOCK*k+2], wires=[curr_wire, wire_idx(i, (j+1)%Ly)])

        return qml.expval(qml.PauliY(0) @ qml.PauliY(1))
    
    return circuit

def sample_param(rng, *, constant, num_qubits, num_blocks):
    num_params = num_blocks * LAYERS_PER_BLOCK
    init_params = rng.random(size = num_params)

    for i in range(num_blocks):
        layer_param_sum = np.sum(init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK])
        init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK] *= (2*constant*math.pi)/(num_qubits*layer_param_sum)

    return init_params

def only_one_is_true(*l):
    return l.count(True) == 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient scaling of hamiltonian variational Ansatz for the XXZ model')
    parser.add_argument('--device', required=True, type=str, choices=["lightning.qubit", "lightning.gpu"], help='Device')
    parser.add_argument('--lattice', required=True, type=str, help='The size of the square lattice')
    parser.add_argument('--num_blocks', required=True, type=int, help='Number of blocks (p)')
    parser.add_argument('--constant', type=float, help='constant')
    parser.add_argument('--num_iter', required=True, type=int, help='Number of total iteration')
    parser.add_argument('--random', action='store_true', help="Use random initialization")
    parser.add_argument('--small', type=float, help="Use small initialization")

    args = parser.parse_args()

    if not only_one_is_true(bool(args.random), bool(args.constant), bool(args.small)):
        raise ValueError("Only one of the arguments --constant, --random, and --small must be given")

    device = args.device
    num_blocks = args.num_blocks
    constant = args.constant
    num_iter = args.num_iter

    lattice = args.lattice.split('x')
    Lx = int(lattice[0])
    Ly = int(lattice[1])

    num_qubits = Lx * Ly
    num_params = num_blocks * LAYERS_PER_BLOCK

    args_in = vars(args)
    with open('args_in.json', 'w') as f:
        json.dump(args_in, f, indent=4)

    rng = Generator(PCG64())

    dev = qml.device(device, wires=num_qubits)
    circuit = qml.QNode(create_circuit(Lx, Ly, num_blocks), dev, diff_method="adjoint")

    grads = np.zeros((num_iter, num_blocks * LAYERS_PER_BLOCK), dtype=np.float128)

    for i in range(num_iter):
        if args.constant:
            param = pnp.array(sample_param(rng, constant=constant, num_qubits=num_qubits, num_blocks=num_blocks), requires_grad=True)
        elif args.random:
            param = pnp.array(2*math.pi*rng.random(size = num_params), requires_grad=True)
        elif args.small:
            param = pnp.array(args.small*rng.random(size = num_params), requires_grad=True)
        grad = qml.grad(circuit)(param)
        grads[i,:] = grad

    np.save("grads.npy", grads)
