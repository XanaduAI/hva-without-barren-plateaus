import numpy as np
import math

def create_neel_st(num_qubits):
    assert num_qubits % 2 == 0
    st_01 = np.array([0.0, 1.0, 0.0, 0.0]) # = kron([1,0],[0,1])
    st_10 = np.array([0.0, 0.0, 1.0, 0.0]) # = kron([0,1],[1,0])
    st_01_N_2 = np.array([1.0])
    st_10_N_2 = np.array([1.0])

    for i in range(num_qubits // 2):
        st_01_N_2 = np.kron(st_01_N_2, st_01)
        st_10_N_2 = np.kron(st_10_N_2, st_10)

    return (st_01_N_2 + st_10_N_2) / math.sqrt(2)

def create_neel_st_2d(Lx, Ly):
    assert Lx % 2 == 0

    st_01 = np.array([0.0, 1.0, 0.0, 0.0]) # = kron([1,0],[0,1])
    st_10 = np.array([0.0, 0.0, 1.0, 0.0]) # = kron([0,1],[1,0])
    st_01_Lx = np.array([1.0])
    st_10_Lx = np.array([1.0])

    for i in range(Lx // 2):
        st_01_Lx = np.kron(st_01_Lx, st_01)
        st_10_Lx = np.kron(st_10_Lx, st_10)

    st_neel0 = np.array([1.0])
    st_neel1 = np.array([1.0])

    for j in range(Ly):
        if j % 2 == 0:
            st_neel0 = np.kron(st_neel0, st_01_Lx)
            st_neel1 = np.kron(st_neel1, st_10_Lx)
        else:
            st_neel0 = np.kron(st_neel0, st_10_Lx)
            st_neel1 = np.kron(st_neel1, st_01_Lx)

    return (st_neel0 + st_neel1) / math.sqrt(2)

