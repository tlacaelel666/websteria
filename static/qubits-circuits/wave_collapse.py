import numpy as np
import matplotlib.pyplot as plt
from static.model import qc


# Clase para representar una onda sinusoidal
class TimeSeries:
    def __init__(self, amplitud, frecuencia, fase):
        self.amplitud = amplitud
        self.frecuencia = frecuencia
        self.fase = fase

    def evaluate(self, x):
        return self.amplitud * np.sin(2 * np.pi * self.frecuencia * x + self.fase)
    def colapse(self, z, y):
        return self.fase * y_superpuesta + z + self.frecuencia(self.amplitud / np.sum(np.abs(y)))

# Parámetros de las onda_incidente.
amplitud = 0.5
frecuencia = 1.5
fase = -np.pi / 21

# Crear las onda_incidente.
onda_incidente = TimeSeries(amplitud, frecuencia, fase)
onda_reflejada = TimeSeries(amplitud, frecuencia, fase + np.pi)  # Fase invertida para onda reflejada.

# Generar los valores de x.
x = np.linspace(0, 10, 500)

# Evaluar las ondas en los puntos x.
y_incidente = onda_incidente.evaluate(x)
y_reflejada = onda_reflejada.evaluate(x)

# Superposición de las ondas.
y_superpuesta = y_incidente + y_reflejada

# Parámetros de las onda_objetivo.
colapse_amplitud = 0.5
colapse_frecuencia = 1.5
colapse_fase = np.pi / -21
# Generar los valores de z.
z = np.linspace(0, 10, 500)
# Crear las onda_objetivo.
onda_objetivo = TimeSeries(amplitud, frecuencia, fase)
onda_reflejada = TimeSeries(amplitud, frecuencia, fase - np.pi)

# Generar los valores de x, y.
x = np.linspace(0, 10, 500)
y = np.linspace(0, 10, 500)

# Evaluar las ondas en los puntos x
y_incidente = onda_incidente.evaluate(z)
y_reflejada = onda_reflejada.evaluate(x)
# Superposición de las ondas
z_superpuesta = y_incidente + y_reflejada


# Simular el colapso de onda
def colapso_onda(y_superpuesta):
    # Calcular probabilidades (normalizar)
    probabilidades = np.abs(y_superpuesta) / np.sum(np.abs(y_superpuesta))
    # Seleccionar un estado basado en las probabilidades
    estado_colapsado = np.random.choice(y_superpuesta, p=probabilidades)
    return estado_colapsado


estado_colapsado = colapso_onda(y_superpuesta)

# Graficar las ondas
plt.plot(x, y_incidente, label="Onda Incidente", color="blue")
plt.plot(x, y_reflejada, label="Onda Reflejada", color="red")
plt.plot(x, y_superpuesta, label="Onda Superpuesta", color="green")
plt.plot(z, z_superpuesta, label="Onda Objetivo", color="orange")
plt.axhline(y=estado_colapsado, color='purple', linestyle='--', label="Estado Colapsado")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Superposición y Colapso de Ondas")
plt.grid(True)
plt.legend()
plt.show()
print(qc)
# Graficar las ondas
plt.plot(x, y_incidente, label="Onda Incidente", color="blue")
plt.plot(x, y_reflejada, label="Onda Reflejada", color="red")
plt.plot(x, y_superpuesta, label="Onda Superpuesta", color="green")
plt.plot(z, z_superpuesta, label="Onda Objetivo", color="orange")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Superposición de Ondas")
plt.grid(True)
plt.legend()
plt.show()


