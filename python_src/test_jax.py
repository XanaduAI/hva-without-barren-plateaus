import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import jax
import jax.numpy as jnp
import math
import sys
from numpy.random import Generator, PCG64
import argparse
import optax
from catalyst import qjit, measure, cond, for_loop, while_loop, grad
from utils import create_neel_st

import time

NUM_WIRES = 10
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
    
if __name__ == '__main__':
    total_steps = 10

    eta = 0.025
    num_shots = 100

    params = math.pi*jnp.ones((NUM_PARAMS,))
    dev = qml.device('default.qubit.jax', wires=NUM_WIRES, shots=num_shots)
    qnode = qml.QNode(circuit, dev, diff_method="parameter-shift", interface="jax")

    optimizer = optax.adam(learning_rate = eta, b1=0.9, b2=0.999, eps=1e-7)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        cost, grads = jax.value_and_grad(qnode)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, cost

    t0 = time.time()

    for iter in range(total_steps):
        params, opt_state, cost = step(params, opt_state)
        print(f"step: {iter}, cost: {cost}", flush=True)

    t1 = time.time()
    print(t1 - t0)
