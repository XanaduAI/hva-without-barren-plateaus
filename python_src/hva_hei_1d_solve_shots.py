import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import jax
import jax.numpy as jnp
import math
import sys
from numpy.random import Generator, PCG64
import argparse
from utils import create_neel_st
import optax

NUM_WIRES = 12
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
# ham_mat = make_xxz_ham(1.0).sparse_matrix()
# ham = qml.SparseHamiltonian(ham_mat, wires=range(NUM_WIRES))
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

    # return qml.expval(ham)
    return qml.expval(qml.PauliZ(0))
    
if __name__ == '__main__':
    total_steps = 10

    parser = argparse.ArgumentParser(description='Run VQE for solving the 1D Heisenberg model')
    parser.add_argument('--learning_rate', required=True, help='Learning rate. Values bettwen [0.005, 0.5] works well', type=float)
    parser.add_argument('--param_init', choices=['random', 'constraint', 'pi'], required=True, help='Parameter initialization scheme')
    parser.add_argument('--num_shots', required=True, help='Number of shots', type=int)

    args = parser.parse_args()
    eta = args.learning_rate
    param_init = args.param_init
    num_shots = args.num_shots

    print(f'#Learning: {eta}')
    print(f'#Parameter Initialize: {param_init}')

    rng = Generator(PCG64())
    if param_init == 'pi':
        init_params = math.pi*np.ones((NUM_PARAMS,))
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
    dev = qml.device('default.qubit.jax', wires=NUM_WIRES, shots=num_shots)
    params = jnp.array(init_params)

    qnode = qml.QNode(circuit, dev, interface="jax", diff_method="parameter-shift")

    optimizer = optax.adam(learning_rate = eta, b1=0.9, b2=0.999, eps=1e-7)
    opt_state = optimizer.init(params)

    for step in range(total_steps):
        cost, grads = jax.value_and_grad(qnode)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"step: {step}, cost: {cost}", flush=True)
