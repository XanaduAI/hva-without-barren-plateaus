import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
from utils import create_neel_st

NUM_WIRES = 14
TOTAL_BLOCKS = NUM_WIRES
LAYERS_PER_BLOCK = 3
NUM_PARAMS = TOTAL_BLOCKS * LAYERS_PER_BLOCK

def make_xxz_ham(delta):
    obs = []
    coeffs = []
    for w in range(NUM_WIRES):
        obs.append(qml.PauliX(w) @ qml.PauliX((w+1) % NUM_WIRES))
        coeffs.append(1.0)
    for w in range(NUM_WIRES):
        obs.append(qml.PauliY(w) @ qml.PauliY((w+1) % NUM_WIRES))
        coeffs.append(1.0)
    for w in range(NUM_WIRES):
        obs.append(qml.PauliZ(w) @ qml.PauliZ((w+1) % NUM_WIRES))
        coeffs.append(delta)

    return qml.Hamiltonian(coeffs, obs, grouping_type='qwc')

ini_st = create_neel_st(NUM_WIRES)
ham = make_xxz_ham(1.0)

def circuit(x):
    qml.QubitStateVector(ini_st, wires = range(NUM_WIRES))
    for k in range(TOTAL_BLOCKS):
        for w in range(NUM_WIRES):
            qml.IsingXX(2*x[LAYERS_PER_BLOCK*k+0], wires=[w, (w+1) % NUM_WIRES])
        for w in range(NUM_WIRES):
            qml.IsingYY(2*x[LAYERS_PER_BLOCK*k+1], wires=[w, (w+1) % NUM_WIRES])
        for w in range(NUM_WIRES):
            qml.IsingZZ(2*x[LAYERS_PER_BLOCK*k+2], wires=[w, (w+1) % NUM_WIRES])

    return qml.expval(ham)

def calc_grad(x):
    return qml.grad(circuit)(x)
    
if __name__ == '__main__':
    #total_steps = 10
    total_steps = 200

    parser = argparse.ArgumentParser(description='Run VQE for solving the 1D Heisenberg model')
    parser.add_argument('--learning_rate', required=True, help='Learning rate. Values bettwen [0.005, 0.5] works well', type=float)
    parser.add_argument('--param_init', choices=['random', 'constraint', 'pi'], required=True, help='Parameter initialization scheme')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots')

    args = parser.parse_args()
    eta = args.learning_rate
    param_init = args.param_init
    num_shots = args.num_shots

    print(f'#Learning: {eta}')
    print(f'#Parameter Initialize: {param_init}')

    rng = Generator(PCG64())
    if param_init == 'pi':
        init_params = math.pi*np.ones((NUM_PARAMS, ))
    elif param_init == 'random':
        low = 0.0
        high = 2*math.pi
        init_params = (high - low)*rng.random(size = NUM_PARAMS) + low
    elif param_init == 'constraint':
        low = 0.0
        high = 2*math.pi
        init_params = (high - low)*rng.random(size = NUM_PARAMS) + low
        for i in range(NUM_PARAMS // LAYERS_PER_BLOCK):
            layer_param_sum = np.sum(init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK])
            init_params[i*LAYERS_PER_BLOCK:(i+1)*LAYERS_PER_BLOCK] *= math.pi/(2*NUM_WIRES)/layer_param_sum
    else:
        raise ValueError('Unkown command line option value for --param_init')

    print(init_params, file=sys.stderr)

    dev = qml.device('lightning.qubit', wires=NUM_WIRES, shots=num_shots)
    qnode = qml.QNode(circuit, dev, diff_method="parameter-shift", interface="autograd")
    params = pnp.array(init_params, requires_grad = True)

    opt = qml.AdamOptimizer(eta, beta1=0.9, beta2=0.999, eps=1e-7)

    for step in range(total_steps):
        cost = qnode(params)
        params = opt.step(qnode, params)
        print(f"step: {step}, cost: {cost}", flush=True)
