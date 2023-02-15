PYTHON3 := $(shell which python3 2> /dev/null)


.PHONY: clean

clean:
	$(PYTHON3) setup.py clean --all
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf ./build ./PauliHamiltonianBuilder.egg-info
	rm -rf ./pauli_hamiltonian_builder/pauli_hamiltonian_builder*
