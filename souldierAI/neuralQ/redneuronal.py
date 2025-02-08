import numpy as np

from typing import Tuple, List, Dict

import pickle

# Funciones de activación y derivadas
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, activation: str = "sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Inicializar pesos y sesgos
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
        deltas = [output_error * self.activate_derivative(activations[-1])]
        
        # Calculando los deltas para cada capa
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * self.activate_derivative(activations[i + 1])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Actualizar pesos y sesgos usando el optimizador seleccionado
        for i in range(len(self.weights)):
            if optimizer == "sgd":
                self.weights[i] += activations[i].T.dot(deltas[i]) * learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
            elif optimizer == "adam":
                self._adam(i, activations[i], deltas[i], learning_rate, **kwargs)
            else:
                raise ValueError("Optimizador no reconocido.")
    
    def _adam(self, layer: int, a: np.ndarray, delta: np.ndarray, learning_rate: float, t: int,
              beta1=0.9, beta2=0.999, epsilon=1e-8, m_w=None, v_w=None, m_b=None, v_b=None):
        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        
        self.m_w[layer] = beta1 * self.m_w[layer] + (1 - beta1) * a.T.dot(delta)
        self.v_w[layer] = beta2 * self.v_w[layer] + (1 - beta2) * np.square(a.T.dot(delta))
        
        m_hat_w = self.m_w[layer] / (1 - beta1**t)
        v_hat_w = self.v_w[layer] / (1 - beta2**t)
        
        self.weights[layer] += learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
        
        self.m_b[layer] = beta1 * self.m_b[layer] + (1 - beta1) * np.sum(delta, axis=0, keepdims=True)
        self.v_b[layer] = beta2 * self.v_b[layer] + (1 - beta2) * np.square(np.sum(delta, axis=0, keepdims=True))
        
        m_hat_b = self.m_b[layer] / (1 - beta1**t)
        v_hat_b = self.v_b[layer] / (1 - beta2**t)
        
        self.biases[layer] += learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

class DataProcessor:
    def __init__(self):
        pass
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def split_data(self, data: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.permutation(len(data))
        test_size = int(test_size * len(data))
        test_idx, train_idx = indices[:test_size], indices[test_size:]
        
        return data[train_idx], data[test_idx], labels[train_idx], labels[test_idx]
    
    def k_fold_split(self, data: np.ndarray, labels: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
    
    def calculate_accuracy(self, output: np.ndarray, labels: np.ndarray) -> float:
        predictions = (output > 0.5).astype(int)
        return np.mean(predictions == labels)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1, optimizer: str = "sgd") -> Dict[str, List[float]]:
        for epoch in range(1, epochs + 1):
            activations = self.model.forward(X)
            output = activations[-1]
            
            loss = np.mean(np.square(y - output))
            accuracy = self.calculate_accuracy(output, y)
            
            self.model.backward(X, y, learning_rate=learning_rate, optimizer=optimizer, t=epoch)
            
            # Guardar el historial de pérdida y precisión
            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
        
        return self.history
    
    def cross_validate(self, data: np.ndarray, labels: np.ndarray, k: int, epochs: int, learning_rate: float, optimizer: str = "sgd"):
        folds = self.processor.k_fold_split(data, labels, k)
        avg_loss = 0
        avg_accuracy = 0
        
        for i, (train_X, test_X, train_y, test_y) in enumerate(folds):
            print(f"Fold {i + 1}/{k}")
            self.train(train_X, train_y, epochs=epochs, learning_rate=learning_rate, optimizer=optimizer)
            _, output = self.model.forward(test_X)
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

# Ejemplo de uso
if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    processor = DataProcessor()
    neural_network = NeuralNetwork(input_size=3, hidden_size=[4, 3], output_size=1, activation="relu")
    trainer = ModelTrainer(neural_network, processor)
    
    X_normalized = processor.normalize_data(X)
    
    history = trainer.train(X_normalized, y, epochs=1000, learning_rate=0.01, optimizer="adam")
    trainer.save_model('neural_network_model.pkl')
    
    # Cargar el modelo
trainer.load_model('neural_network_model.pkl')
predictions = neural_network.forward(X_normalized)
    
print("\nPredicciones finales:")
print(predictions)
