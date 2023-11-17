import pennylane as qml
import numpy as np
import math
import sys
from numpy.random import Generator, PCG64
import argparse
from utils import create_neel_st_2d
from hva_xyz import HVA
import jax.numpy as jnp
import optax
from tqdm import tqdm
import os

Lx = 4
Ly = 4
NUM_WIRES = Lx*Ly
TOTAL_BLOCKS = NUM_WIRES
LAYERS_PER_BLOCK = 3
NUM_PARAMS = TOTAL_BLOCKS * LAYERS_PER_BLOCK

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def wire_idx(i, j):
    return j*Lx + i

if __name__ == '__main__':
    total_steps = 20
    #total_steps = 1000

    parser = argparse.ArgumentParser(description='Run VQE for solving the 2D Heisenberg model with shots')
    parser.add_argument('--learning_rate', required=True, help='Learning rate. Values bettwen [0.005, 0.5] works well', type=float)
    parser.add_argument('--param_init', choices=['random', 'constraint', 'pi'], required=True, help='Parameter initialization scheme')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots')

    args = parser.parse_args()
    eta = args.learning_rate
    param_init = args.param_init
    num_shots = args.num_shots

    print(f'#Learning: {eta}')
    print(f'#Parameter initialize: {param_init}')
    print(f'#Number of shots: {num_shots}')

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

    params = jnp.array(init_params)

    optimizer = optax.adam(eta, b2=0.999, eps=1e-7)
    opt_state = optimizer.init(params)

    edges = []
    for i in range(Lx):
        for j in range(Ly):
            curr_wire = wire_idx(i, j)
            edges.append([curr_wire, wire_idx((i+1)%Lx, j)])
            edges.append([curr_wire, wire_idx(i, (j+1)%Ly)])

    hva_xyz = HVA(NUM_WIRES, TOTAL_BLOCKS, edges)

    ini_st = create_neel_st_2d(Lx, Ly)

    for step in tqdm(range(total_steps)):
        grads = hva_xyz.grad_shots(ini_st, params, num_shots)
        grads = jnp.array(grads)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 10 == 0:
            cost = hva_xyz.expval(ini_st, params)
            print(f"step: {step}, cost: {cost:.8f}", flush=True)
