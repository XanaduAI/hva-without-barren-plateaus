import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
from utils import create_neel_st_2d
from datetime import datetime

Lx = 6
Ly = 4
NUM_WIRES = Lx*Ly
TOTAL_BLOCKS = NUM_WIRES
LAYERS_PER_BLOCK = 3
NUM_PARAMS = TOTAL_BLOCKS * LAYERS_PER_BLOCK

def wire_idx(i, j):
    return j*Lx + i

dev = qml.device('lightning.gpu', wires=NUM_WIRES)

def make_xxz_ham(delta):
    obs = []
    coeffs = []

    for i in range(Lx):
        for j in range(Ly):
            curr_wire = wire_idx(i, j)
            obs.append(qml.PauliX(curr_wire) @ qml.PauliX(wire_idx((i+1)%Lx, j)))
            obs.append(qml.PauliX(curr_wire) @ qml.PauliX(wire_idx(i, (j+1)%Ly)))
            coeffs.append(1.0)
            coeffs.append(1.0)
    for i in range(Lx):
        for j in range(Ly):
            curr_wire = wire_idx(i, j)
            obs.append(qml.PauliY(curr_wire) @ qml.PauliY(wire_idx((i+1)%Lx, j)))
            obs.append(qml.PauliY(curr_wire) @ qml.PauliY(wire_idx(i, (j+1)%Ly)))
            coeffs.append(1.0)
            coeffs.append(1.0)
    for i in range(Lx):
        for j in range(Ly):
            curr_wire = wire_idx(i, j)
            obs.append(qml.PauliZ(curr_wire) @ qml.PauliZ(wire_idx((i+1)%Lx, j)))
            obs.append(qml.PauliZ(curr_wire) @ qml.PauliZ(wire_idx(i, (j+1)%Ly)))
            coeffs.append(delta)
            coeffs.append(delta)

    return qml.Hamiltonian(coeffs, obs)

ini_st = create_neel_st_2d(Lx, Ly)

t1 = datetime.now()
ham = qml.SparseHamiltonian(qml.utils.sparse_hamiltonian(make_xxz_ham(1.0)), wires=range(NUM_WIRES))
tdiff = datetime.now() - t1
print("#Time to construct sparse ham: {}s".format(tdiff.total_seconds()))

@qml.qnode(dev, diff_method="adjoint", interface="autograd")
def circuit(x):
    qml.QubitStateVector(ini_st, wires = range(NUM_WIRES))
    for k in range(TOTAL_BLOCKS):
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

    return qml.expval(ham)

def calc_grad(x):
    return qml.grad(circuit)(x)
    
if __name__ == '__main__':
    total_steps = 300

    parser = argparse.ArgumentParser(description='Run VQE for solving the 2D Heisenberg model')
    parser.add_argument('--learning_rate', required=True, help='Learning rate. Values bettwen [0.005, 0.5] works well', type=float)
    parser.add_argument('--param_init', choices=['random', 'constraint', 'pi'], required=True, help='Parameter initialization scheme')

    args = parser.parse_args()
    eta = args.learning_rate
    param_init = args.param_init

    print(f'#Learning: {eta}')
    print(f'#Parameter Initialize: {param_init}')

    opt = qml.AdamOptimizer(eta, beta1=0.9, beta2=0.999, eps=1e-7)
    rng = Generator(PCG64())
    if param_init == 'random':
        low = 0.0
        high = 2*math.pi
        init_params = (high - low)*rng.random(size = NUM_PARAMS) + low
    elif param_init == 'pi':
        init_params = math.pi*np.ones((NUM_PARAMS, 1))
    elif param_init == 'constraint':
        low = 0.0
        high = 2*math.pi
        init_params = (high - low)*rng.random(size = NUM_PARAMS) + low
        for i in range(NUM_PARAMS // LAYERS_PER_BLOCK):
            layer_param_sum = np.sum(init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK])
            init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK] *= math.pi/NUM_WIRES/layer_param_sum
    else:
        raise ValueError('Unkown command line option value for --param_init')

    print(init_params, file=sys.stderr)
    params = pnp.array(init_params, requires_grad = True)

    qnode = circuit

    for step in range(total_steps):
        cost = qnode(params)
        params = opt.step(qnode, params)
        print(f"step: {step}, cost: {cost}", flush=True)
