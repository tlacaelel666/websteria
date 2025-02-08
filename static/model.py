import qiskit
from qiskit.circuit.library.n_local import TwoLocal
from qiskit_aer import Aer
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap  # Importa CouplingMap

# Crear un circuito cuántico con 5 qubits
qc = qiskit.QuantumCircuit(5)

# Definir el mapa de acoplamiento (coupling map)
coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])

# Crear el ansatz TwoLocal usando from_coupling_map
ansatz = TwoLocal(
    rotation_blocks=["rx", "ry", "rz"],
    entanglement="linear",  # o "circular"
    reps=2,
    insert_barriers=True
)

qc.compose(ansatz, inplace=True)

# Medir todos los qubits
qc.measure_all()

# Simulador y transpilación (como en la respuesta anterior)
simulator = Aer.get_backend('qasm_simulator')
transpiled_circuit = qiskit.transpile(qc, simulator)
job = simulator.run(transpiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts(qc)

# --- Aquí comienza la parte de graficación ---

# 1. Preparar los datos para el gráfico (manejar el caso de counts vacío)
if counts:
    x = list(counts.keys())  # Estados resultantes (cadenas binarias)
    y = list(counts.values())  # Frecuencias de cada estado

    # 2. Crear el gráfico de barras
    plt.figure(figsize=(10, 6))  # Ajustar el tamaño del gráfico
    plt.bar(x, y)
    plt.xlabel("Estado Cuántico Resultante")
    plt.ylabel("Frecuencia")
    plt.title("Resultados de la Simulación Cuántica")
    plt.xticks(rotation=45, ha="right")  # Rotar etiquetas del eje x para mejor legibilidad
    plt.tight_layout()  # Ajustar diseño para evitar etiquetas cortadas

    # 3. Mostrar el gráfico
    plt.show()
else:
    print("No se obtuvieron resultados de la simulación.")


# Visualizar el circuito
qc.draw("mpl")

