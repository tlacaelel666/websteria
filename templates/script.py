from flask import Flask, render_template
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
app = Flask(__name__, static_folder='static')

# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Transpile for visualization
qc_transpiled = transpile(qc, basis_gates=["cx", "h", "measure"], optimization_level=3)

def index():
    circuit_json = qc_transpiled.qasm()
    return render_template('index.html', circuit=circuit_json)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
