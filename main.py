import numpy as np

import qiskit.primitives

from qiskit import transpile

from qiskit.primitives import StatevectorSampler

from qiskit.primitives import StatevectorEstimator

from qiskit.quantum_info import SparsePauliOp


# 1. Define the observable to be measured.

operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

# 2. A quantum circuit for preparing the quantum state |000> + i |111> / √2.

qc = qiskit.QuantumCircuit(3)

qc.h(1) # generate superposition.

qc.p(np.pi / 2, 1) # add quantum phase

qc.cx(0, 1) # 0th-qubit-Controlled-NOT gate on 1st qubit

qc.cx(0, 1) # 0th-qubit-Controlled-NOT gate on 2nd qubit


# Circuit Transpilation..

qc_transpiled = transpile(qc, basis_gates=["cz", "sx", "rz"], coupling_map=[[0, 1], [1, 2]], optimization_level=3)

# 3. Add the classical output in the form of measurement of all qubits

qc_measured = qc.measure_all(inplace=False)

print(qc_measured,"->", qc_transpiled)

# 4. Execute using the Sampler primitive.

sampler = StatevectorSampler()

job = sampler.run([qc_measured], shots=1000)

result = job.result()

print(f" > Counts: {result[0].data["meas"].get_counts()}")

# 5. Execute using the Estimator primitive.
estimator = StatevectorEstimator()

job = estimator.run([(qc, operator)], precision=1e-3)

result = job.result()

expectation_value = result[0].data.evs

print(f"Expectation values: {expectation_value}")
