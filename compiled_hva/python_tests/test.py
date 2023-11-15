import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

dev = qml.device('lightning.qubit', wires=16)


def create_circuit(num_blocks, edges):

    ops = []
    coeffs = []

    for edge in edges:
        ops.append(qml.PauliX(edge[0]) @ qml.PauliX(edge[1]))
        ops.append(qml.PauliY(edge[0]) @ qml.PauliY(edge[1]))
        ops.append(qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
        coeffs.append(1.0)
        coeffs.append(1.0)
        coeffs.append(1.0)

    ham = qml.Hamiltonian(coeffs, ops)
        
    @qml.qnode(dev)
    def circuit(params):
        for k in range(num_blocks):
            for edge in edges:
                qml.IsingXX(params[3*k+0], wires=edge)
            for edge in edges:
                qml.IsingYY(params[3*k+1], wires=edge)
            for edge in edges:
                qml.IsingZZ(params[3*k+2], wires=edge)

        return qml.expval(ham)
    return circuit

edges = []
for i in range(16):
    edges.append([i, (i+1)%16])
edges.append([1, 3])

my_circuit = create_circuit(3, edges)
params = pnp.array(0.2*np.ones(9), requires_grad=True)
params[3] = 0.8

print(my_circuit(params))
print(qml.grad(my_circuit)(params))
