import qiskit
from qiskit_aer import Aer

# Crear un circuito cuántico con 3 qubits.
qc = qiskit.QuantumCircuit(3)

# Inicializar el circuito en un estado de superposición.
qc.h(0) # Aplicar Hadamard al primer qubit qc.h(1) # Aplicar Hadamard al segundo qubit.

# Definir el número de iteraciones.
num_iterations = 500

# Simular el "bucle infinito".
""" Aplicar la puerta Toffoli 
    (control: qubit 0 y 1, objetivo: qubit 2) qc.ccx(0, 1, 2) 
     Aplicar la puerta X al qubit auxiliar qc.x(2)."""
for _ in range(num_iterations):
    qc.ccx(0, 1, 2)
    qc.x(2)

# Medir todos los qubits.
qc.measure_all()

# Mostrar el circuito sintetizado.
print(qc.measure_all())

# Crear un simulador para ejecutar el circuito.
sampler = Aer.get_backend('qasm_simulator')

# Compilar el circuito.
compiled_circuit = qiskit.transpile(qc, sampler)
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

max_hamiltonian = SparsePauliOp.from_list([
    ("IIZZ", 1), ("IZIZ", 1), ("IZZI", 1), ("ZIIZ", 1), ("ZZII", 1)
])


max_ansatz = QAOAAnsatz(max_hamiltonian, reps=2)
# Draw
max_ansatz.decompose(reps=3).draw("mpl")
# Ejecutar el circuito en el simulador.
result = sampler.run(compiled_circuit, backend=sampler, shots=1024).result()

#Obtener y mostrar los resultados.
counts = result.get_counts(compiled_circuit)
print("Resultados de la simulación:", counts)
import numpy as np
import matplotlib.pyplot as plt

# --- Example wave function (replace it with a objective function) ---
x = np.linspace(0, 2 * np.pi, 100)  # x-values from 0 to 2*pi
y = np.sin(x)                         # Example: sine wave

# --- Plotting ---
plt.scatter(x, y, c='r') # Corrected: 'c' or 'color' for color
plt.xlabel("x")
plt.ylabel("ψ(x)")  #  ψ is often used to represent a wave function
plt.title("Wave Function")
plt.grid(True)
plt.show()
"""

**Explanation and Improvements:**

1. **`import numpy as np` and `import matplotlib.pyplot as plt`:** Import necessary libraries. NumPy for numerical operations and Matplotlib for plotting.

2. **Define `x` and `y`:** `x` represents the independent variable (e.g., position), and `y` represents the wave function's value at each `x`.  The example uses `np.linspace` to create an array of 100 evenly spaced points between 0 and 2π, and `np.sin(x)` calculates the sine of each `x` value. *Replace this with your actual wave function.*

3. **`plt.scatter(x, y, c='r')`:** This is the core plotting command.
   - `x` and `y` are the data to plot.
   - `c='r'` sets the color to red. You can use other color codes like 'b' (blue), 'g' (green), 'k' (black), etc., or hexadecimal color codes like '#FF0000' (red).  You can also use `color='red'`.

4. **`plt.xlabel("x")`, `plt.ylabel("ψ(x)")`, `plt.title("Wave Function")`:** Add labels and a title for clarity.

5. **`plt.grid(True)`:**  Adds a grid to the plot, making it easier to read values.

6. **`plt.show()`:** Displays the plot.


**Example for a more complex wave function:**"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
y = np.exp(-x**2) * np.cos(2*np.pi*x)  # Example: Gaussian wave packet

plt.plot(x, y, color='blue', label='Wave Function') # plot is better for continuous functions
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Gaussian Wave Packet")
plt.grid(True)
plt.legend() # Show the label
plt.show()
"""
Output
El código crea y simula un circuito cuántico utilizando Qiskit. 
Descripción del Circuito
Inicialización del Circuito:
Se crea un circuito cuántico con 3 qubits.
Se aplican puertas Hadamard (H) a los qubits 0 y 1,
lo que los coloca en un estado de superposición. 
Esto significa que cada uno de estos qubits
está en una superposición de los estados (|0\rangle) y (|1\rangle).

** Bucle de Iteraciones: **
Se define un bucle que se ejecuta 500 veces.
Dentro del bucle, se aplica una puerta Toffoli (CCX),
 que es una puerta de control-control-NOT.
Esta puerta utiliza los qubits 0 y 1 como controles y el qubit 2 como objetivo.
La puerta Toffoli invierte el estado del qubit objetivo
 solo si ambos qubits de control están en el estado (|1\rangle).
Después de la puerta Toffoli, se aplica una puerta X al qubit 2, que invierte su estado.

** Medición: **
Se mide el estado de todos los qubits al final del circuito.

* Simulación
El circuito se ejecuta en un simulador cuántico (qasm_simulator) con 1024 "shots" o repeticiones.
Los resultados de la simulación se obtienen en forma de un diccionario 
que cuenta cuántas veces se midió cada posible estado de los qubits.
* Interpretación del Output
El resultado de la simulación (counts) es un diccionario donde las claves son cadenas de bits 
que representan los estados medidos de los qubits, 
y los valores son el número de veces que se midió cada estado.

Dado que el circuito aplica
una puerta Toffoli seguida de una puerta X al qubit 2 en cada iteración, 
el estado del qubit 2 se alterna en cada iteración.
 Sin embargo, debido a que el circuito se mide al final,
lo que realmente se observa es el efecto acumulado de todas las operaciones.
El estado final de los qubits dependerá de cómo las puertas Toffoli y X interactúan
a lo largo de las iteraciones.

 En este caso, debido a la naturaleza del bucle y las puertas aplicadas,
En el resultado típicamente verás una distribución de estados 
que refleja la interferencia cuántica y las operaciones realizadas.

Por ejemplo, podrías ver resultados como {'000': 512, '111': 512} 
si el circuito efectivamente alterna entre dos estados principales,
 esto es solo un ejemplo hipotético.
El resultado real dependerá de la implementación y el comportamiento del circuito 
bajo las operaciones definidas."""""