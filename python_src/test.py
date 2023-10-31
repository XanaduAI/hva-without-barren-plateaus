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

NUM_WIRES = 10
TOTAL_BLOCKS = NUM_WIRES
LAYERS_PER_BLOCK = 3
NUM_PARAMS = TOTAL_BLOCKS * LAYERS_PER_BLOCK

def circuit(x):
    for k in range(TOTAL_BLOCKS):
        for w in range(NUM_WIRES):
            qml.IsingXX(2*x[LAYERS_PER_BLOCK*k+0], wires=[w, (w+1) % NUM_WIRES])
        for w in range(NUM_WIRES):
            qml.IsingYY(2*x[LAYERS_PER_BLOCK*k+1], wires=[w, (w+1) % NUM_WIRES])
        for w in range(NUM_WIRES):
            qml.IsingZZ(2*x[LAYERS_PER_BLOCK*k+2], wires=[w, (w+1) % NUM_WIRES])

    return qml.expval(qml.PauliZ(0))
    
if __name__ == '__main__':
    total_steps = 10

    eta = 0.01
    num_shots = 10

    dev = qml.device('default.qubit.jax', wires=NUM_WIRES, shots=num_shots)
    params = math.pi*jnp.ones((NUM_PARAMS,))

    qnode = qml.QNode(circuit, dev, interface="jax", diff_method="parameter-shift")

    optimizer = optax.adam(learning_rate = eta, b1=0.9, b2=0.999, eps=1e-7)
    opt_state = optimizer.init(params)

    for step in range(total_steps):
        print(step, flush=True)
        cost, grads = jax.value_and_grad(qnode)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"step: {step}, cost: {cost}", flush=True)
