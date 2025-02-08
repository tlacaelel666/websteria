import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List, Tuple, Any

from numpy import floating


# --- Activation Functions (From redneuronal.py) ---
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


# --- Neural Network Components (From redneuronal.py)---
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, activation: str = "sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []

        previous_size = input_size
        for hs in hidden_size:
            self.weights.append(np.random.randn(previous_size, hs))
            self.biases.append(np.zeros((1, hs)))
            previous_size = hs
        self.weights.append(np.random.randn(previous_size, output_size))
        self.biases.append(np.zeros((1, output_size)))

    def activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)
        elif self.activation == "relu":
            return relu(x)
        else:
            raise ValueError("Función de activación no reconocida.")

    def activate_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "tanh":
            return tanh_derivative(x)
        elif self.activation == "relu":
            return relu_derivative(x)
        else:
            raise ValueError("Función de activación no reconocida.")

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.activate(np.dot(X, w) + b)
            activations.append(X)
        return activations

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, optimizer: str = "sgd", **kwargs):
        activations = self.forward(X)
        output = activations[-1]
        output_error = y - output
        deltas = [output_error * self.activate_derivative(output)]

        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * self.activate_derivative(activations[i + 1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            if optimizer == "sgd":
                self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            elif optimizer == "adam":
                self._adam(i, activations[i], deltas[i], learning_rate, **kwargs)
            else:
                raise ValueError("Optimizador no reconocido.")

    def _adam(self, layer: int, a: np.ndarray, delta: np.ndarray, learning_rate: float, t: int,
              beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]

        # Update biased first moment estimate
        self.m_w[layer] = beta1 * self.m_w[layer] + (1 - beta1) * np.dot(a.T, delta)
        self.m_b[layer] = beta1 * self.m_b[layer] + (1 - beta1) * np.sum(delta, axis=0, keepdims=True)

        # Update biased second raw moment estimate
        self.v_w[layer] = beta2 * self.v_w[layer] + (1 - beta2) * np.dot(a.T, delta) ** 2
        self.v_b[layer] = beta2 * self.v_b[layer] + (1 - beta2) * (np.sum(delta, axis=0, keepdims=True) ** 2)

        # Compute bias-corrected first moment estimate
        m_hat_w = self.m_w[layer] / (1 - beta1 ** t)
        m_hat_b = self.m_b[layer] / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_hat_w = self.v_w[layer] / (1 - beta2 ** t)
        v_hat_b = self.v_b[layer] / (1 - beta2 ** t)

        # Update parameters
        self.weights[layer] += learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
        self.biases[layer] += learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)


class DataProcessor:
    def __init__(self):
        pass

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def split_data(self, data: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.permutation(len(data))
        test_size = int(test_size * len(data))
        test_idx, train_idx = indices[:test_size], indices[test_size:]

        return data[train_idx], data[test_idx], labels[train_idx], labels[test_idx]

    def k_fold_split(self, data: np.ndarray, labels: np.ndarray, k: int) -> List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        fold_size = len(data) // k
        indices = np.random.permutation(len(data))
        folds = []

        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
            folds.append((data[train_idx], data[test_idx], labels[train_idx], labels[test_idx]))

        return folds


class ModelTrainer:
    def __init__(self, model: NeuralNetwork, processor: DataProcessor):
        self.model = model
        self.processor = processor
        self.history = {'loss': [], 'accuracy': []}

    def calculate_accuracy(self, output: np.ndarray, labels: np.ndarray) -> floating[Any]:
        predictions = (output > 0.5).astype(int)
        return np.mean(predictions == labels)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1,
              optimizer: str = "sgd") -> dict:
        for epoch in range(1, epochs + 1):
            activations = self.model.forward(X)
            output = activations[-1]

            loss = np.mean(np.square(y - output))
            accuracy = self.calculate_accuracy(output, y)

            self.model.backward(X, y, learning_rate=learning_rate, optimizer=optimizer, t=epoch)

            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

        return self.history

    def cross_validate(self, data: np.ndarray, labels: np.ndarray, k: int, epochs: int, learning_rate: float,
                       optimizer: str = "sgd"):
        folds = self.processor.k_fold_split(data, labels, k)
        avg_loss = 0
        avg_accuracy = 0

        for i, (train_X, test_X, train_y, test_y) in enumerate(folds):
            print(f"Fold {i + 1}/{k}")
            self.train(train_X, train_y, epochs=epochs, learning_rate=learning_rate, optimizer=optimizer)
            activations = self.model.forward(test_X)
            output = activations[-1]
            loss = np.mean(np.square(test_y - output))
            accuracy = self.calculate_accuracy(output, test_y)

            avg_loss += loss
            avg_accuracy += accuracy

        avg_loss /= k
        avg_accuracy /= k
        print(f"Cross-validation completed. Avg Loss: {avg_loss}, Avg Accuracy: {avg_accuracy}")

    def save_model(self, filename: str):
        with open(filename, 'wb') as file:
             pickle.dump(self.model, file)

    def load_model(self, filename: str):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)


# --- Quantum State (From quantum.py) ---
class QuantumState:
    def __init__(self, num_positions: int):
        self.num_positions = num_positions
        self.angles = np.linspace(0, np.pi, num_positions)
        self.probabilities = self.calculate_probabilities()
        self.history = [self.probabilities.copy()]

    def calculate_probabilities(self) -> np.ndarray:
        cosines = np.cos(self.angles)
        probabilities = cosines ** 2
        return probabilities / np.sum(probabilities)

    def update_probabilities(self, action: int, k: float = 0.1):
        new_probabilities = self.probabilities.copy()
        pos = self.observe_position()
        p = self.probabilities[pos]

        for i in range(self.num_positions):
            if action == 1:
                if i > pos:
                    new_probabilities[i] += k * p
                elif i < pos:
                    new_probabilities[i] -= k * p
                else:
                    new_probabilities[i] += (self.num_positions - 1) * k * p
            elif action == 0:
                if i < pos:
                    new_probabilities[i] += k * p
                elif i > pos:
                    new_probabilities[i] -= k * p
                else:
                    new_probabilities[i] += (self.num_positions - 1) * k * p
            else:
                raise ValueError("Acción no válida. Debe ser 0 o 1.")

        # Asegurar que las probabilidades no sean negativas antes de normalizar
        new_probabilities = np.clip(new_probabilities, a_min=0, a_max=None)

        # Normalizar las probabilidades
        new_probabilities = new_probabilities / np.sum(new_probabilities)
        self.probabilities = new_probabilities
        self.history.append(new_probabilities.copy())

    def observe_position(self) -> int:
        return np.random.choice(self.num_positions, p=self.probabilities)

    def get_probabilities(self) -> np.ndarray:
        return self.probabilities

    def plot_probabilities(self):
        plt.figure(figsize=(10, 6))
        history_array = np.array(self.history)
        for i in range(self.num_positions):
            plt.plot(range(len(history_array)), history_array[:, i], label=f'Position {i}')
        plt.xlabel('Step')
        plt.ylabel('Probability')
        plt.title('Probability Evolution Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- Quantum Neuron Class ---
class QuantumNeuron:
    def __init__(self, num_positions: int, hidden_size: List[int], output_size: int, activation: str = "relu"):
        self.quantum_state = QuantumState(num_positions)
        self.neural_network = NeuralNetwork(input_size=num_positions, hidden_size=hidden_size, output_size=output_size,
                                            activation=activation)
        self.processor = DataProcessor()
        self.trainer = ModelTrainer(self.neural_network, self.processor)

    def forward(self, action: int) -> np.ndarray:
        """
        Performs forward pass by updating the quantum state based on the action and then passing the current probabilities to the neural network.
        """
        self.quantum_state.update_probabilities(action)
        quantum_probabilities = self.quantum_state.get_probabilities().reshape(1, -1)  # Reshape to fit the NN Input

        return self.neural_network.forward(quantum_probabilities)[-1]  # Return the output of the last layer of the NN

    def train(self, actions: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float,
              optimizer: str = "adam") -> dict:
        """
        Trains the quantum neuron using actions and labels.
        The actions will update the quantum state, and the probabilities will be used as input to the neural network.
        """
        probabilities_history = []
        for action in actions:
            self.quantum_state.update_probabilities(action)
            probabilities_history.append(self.quantum_state.get_probabilities())
        probabilities_history = np.array(probabilities_history).reshape(-1, self.quantum_state.num_positions)

        return self.trainer.train(probabilities_history, labels, epochs, learning_rate, optimizer)

    def plot_probabilities(self):
        self.quantum_state.plot_probabilities()

    def plot_loss_accuracy(self):
        loss_history = self.trainer.history['loss']
        accuracy_history = self.trainer.history['accuracy']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(loss_history)), loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(accuracy_history)), accuracy_history, label='Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Ejemplo de parámetros
    num_positions = 5
    hidden_size = [4, 3]
    output_size = 1
    epochs = 1000
    learning_rate = 0.01

    # Crear el quantum neuron
    quantum_neuron = QuantumNeuron(num_positions, hidden_size, output_size)

    # Definir Acciones y Etiquetas
    actions = np.random.randint(0, 2, epochs)
    labels = np.random.randint(0, 2, (epochs, 1)).astype(
        float)  # Asegurar que las etiquetas sean floats para compatibilidad con NN

    # Entrenar el neurón
    history = quantum_neuron.train(actions, labels, epochs, learning_rate, optimizer="adam")

    # Graficar
    quantum_neuron.plot_probabilities()
    quantum_neuron.plot_loss_accuracy()

    # Ejemplo de pasada hacia adelante
    last_action = np.random.randint(0, 2)
    prediction = quantum_neuron.forward(last_action)
    print("Final Predictions", prediction)


