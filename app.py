# app.py

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

from main import estimator, operator, qc_measured, qc_transpiled, sampler
from static.model import qc


def is_active(node, threshold=0.5):
    """
    Checks if a node is active based on the probability of exposed states.

    Args:
        node (np.ndarray): The node's state matrix.
        threshold (float): The activation threshold.

    Returns:
        bool: True if the node is active, False otherwise.
    """
    prob_exposed = sum(abs(node[i, j]) ** 2 for i in range(node.shape[0]) for j in range(node.shape[1]))
    return prob_exposed > threshold


def visualize_network(network, iteration):
    """
    Visualizes the network state using matplotlib.

    Args:
        network (list of list of np.ndarray): The network to visualize.
        iteration (int): The current iteration number.
    """
    plt.figure(figsize=(8, 6))
    for layer_index, layer in enumerate(network):
        for node_index, node in enumerate(layer):
            if is_active(node):
                plt.scatter(layer_index, node_index, color='red', marker='o')

    plt.title(f"Network State at Iteration {iteration}")
    plt.xlabel("Layer Index")
    plt.ylabel("Node Index")
    plt.xlim(-1, len(network))
    plt.ylim(-1, max(len(layer) for layer in network))
    plt.grid()
    plt.show()


def main():
    # 1. Define the observable to be measured
    operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

    # 2. Create a quantum circuit for preparing the quantum state |000> + i |111> / √2
    qc = QuantumCircuit(3)
    qc.h(0)  # Generate superposition on qubit 0
    qc.p(np.pi / 2, 2)  # Add quantum phase to qubit 2
    qc.cx(1, 0)  # Controlled-NOT gate with qubit 1 as control and qubit 0 as target
    qc.cx(0, 1)  # Controlled-NOT gate with qubit 0 as control and qubit 1 as target

    # Transpile the quantum circuit with specified basis gates and coupling map
    try:
        qc_transpiled = transpile(
            qc,
            basis_gates=["cz", "sx", "rz"],
            coupling_map=[[0, 1], [1, 2]],
            optimization_level=3
        )
    except Exception as e:
        print(f"Error during transpilation: {e}")
        return

    # 3. Add classical measurements to all qubits
    qc_measured = qc.measure_all(inplace=False)

    # 4. Initialize the Sampler and Estimator primitives
    sampler = Aer.get_backend('qasm_simulator')
    compiled_circuit = qiskit.transpile(qc, sampler)
    # Execute the sampler to get counts
try:
    sampler_job = sampler.run(circuits=[qc_measured], shots=1000)
    sampler_result = sampler_job.result()
    counts = sampler_result.result[0].data["counts"]
    print(f" > Counts: {counts}")
except Exception as e:
    print(f"Error during sampling: {e}")
    # Execute the estimator to get expectation values
    try:
        estimator_job = estimator.run(circuits=[qc], observables=[operator], param_values=[{}])
        estimator_result = estimator_job.result()
        expectation_values = estimator_result.values
        print(f"Expectation values: {expectation_values}")
    except Exception as e:
        print(f"Error during estimation: {e}")

    # Display the circuits and observable
    print("Original Circuit:\n", qc.draw())
    print("\nObservable:\n", operator)
    print("\nMeasured Circuit:\n", qc_measured.draw())
    print("\nTranspiled Circuit:\n", qc_transpiled.draw())

    # --- Visualization (Ejemplo) ---
    # Nota: Necesitas definir cómo se estructura 'network' y 'iteration'.
    # Aquí se proporciona un ejemplo ficticio.
    network = [
        [np.array([[1, 0], [0, 1]])],  # Capa 0 con un nodo
        [np.array([[0.6, 0.8], [0.8, 0.6]])],  # Capa 1 con un nodo activo
        [np.array([[0.3, 0.4], [0.4, 0.3]])]  # Capa 2 con un nodo inactivo
    ]
    iteration = 1000
    visualize_network(network, iteration)


if __name__ == "__main__":
    main()