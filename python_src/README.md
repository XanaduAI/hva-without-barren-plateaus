## Python scripts for the Hamiltonian variational ansatz (HVA)

In this directory, we have python scripts for computing gradients of the HVA. We use the HVA for the one- and two-dimensional XYZ models.

Two scripts `hva_xyz_grad_1d.py` and `hva_xyz_grad_2d.py` compute gradients using the HVA for the one- and two-dimensional XYZ models, respectively. These scripts are used to generate data for Figure 3. For example, you can run a script with:

```bash
$ python3 hva_xyz_grad_1d.py --device lightning.qubit --num_qubits 8 --num_blocks 16 --random --num_iter 1024
```

, which computes gradients of the HVA for the 1D XYZ model for 8 qubits, 16 blocks (`p=16` in a paper), and for 1024 random initial parameters sampled from [0, 2π]. Also, `lightning.qubit` device for PennyLane is used for running the circuit.

For Figure 4, we have used `hva_grad_constant_small.py`. This script computes the gradients for a given number of qubits when the parameters are initialized to small constant values.

```bash
$ python3 hva_grad_constant_small.py --device lightning.qubit --num_qubits 10
```

will generate data for `N=10` (pink curve in the Figure).

We also provide scripts for solving the 1D and 2D Heisenberg models. These scripts (`hva_hei_1d_solve.py` and `hva_hei_2d_solve.py`) are used to generate data for Figure 5. Likewise, one can run a script with, e.g.,

```bash
$ python3 hva_hei_1d_solve.py --learning-rate 0.01 --param-init pi
```

for running the VQE for solving the 1D Heisenberg model. The learning rate `0.01` is used and all parameters are initialized to π.


The last script is `hva_grad_repeat.py` which runs a repeated ansatz. This is used to generate data for Figure 6.
