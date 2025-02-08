import matplotlib.pyplot as plt
import numpy as np


class QuantumNeuron:
    def __init__(self, num_estados):
        self.estado = np.zeros(num_estados, dtype=complex)
class QuantumNetwork:
    def __init__(self, capas):
        self.capas = capas
class QuantumState:
    def __init__(self, num_positions):
        """
        Initialize the QuantumState class with a given number of positions.
        Each position corresponds to an angle, and probabilities are calculated
        based on the squared cosine of these angles.

        Args:
            num_positions (int): Number of positions to distribute angles.
        """
        self.num_positions = num_positions
        self.angles = np.linspace(0, np.pi, num_positions)  # Angles distributed between 0 and π
        self.probabilities = self.calculate_probabilities()  # Probabilities based on cosines
        self.history = [self.probabilities.copy()]  # To track probability updates over time

    def calculate_probabilities(self):
        """
        Calculate initial probabilities using squared cosines of the angles.

        Returns:
            numpy.ndarray: Normalized probabilities for each position.
        """
        cosines = np.cos(self.angles)  # Compute cosine values for each angle
        probabilities = cosines**2  # Square the cosines to get positive values
        return probabilities / np.sum(probabilities)  # Normalize so sum of probabilities is 1

    def update_probabilities(self, action, k=0.1):
        """
        Update the probabilities based on the given action.

        Args:
            action (int): 0 for moving left, 1 for moving right.
            k (float): Scaling factor for probability adjustments.
        """
        new_probabilities = self.probabilities.copy()
        for i in range(self.num_positions):
            if action == 1:  # Action to move right
                if i > self.observe_position():
                    # Increase probability if position is to the right
                    new_probabilities[i] += k * self.probabilities[self.observe_position()]
                elif i < self.observe_position():
                    # Decrease probability if position is to the left
                    new_probabilities[i] -= k * self.probabilities[self.observe_position()]
                else:
                    # Increase probability significantly if it matches the observed position
                    new_probabilities[i] += (self.num_positions - 1) * k * self.probabilities[self.observe_position()]
            elif action == 0:  # Action to move left
                if i < self.observe_position():
                    # Increase probability if position is to the left
                    new_probabilities[i] += k * self.probabilities[self.observe_position()]
                elif i > self.observe_position():
                    # Decrease probability if position is to the right
                    new_probabilities[i] -= k * self.probabilities[self.observe_position()]
                else:
                    # Increase probability significantly if it matches the observed position
                    new_probabilities[i] += (self.num_positions - 1) * k * self.probabilities[self.observe_position()]

        # Normalize probabilities to ensure they sum to 1
        new_probabilities = new_probabilities / np.sum(new_probabilities)

        self.probabilities = new_probabilities  # Update the probabilities
        self.history.append(new_probabilities.copy())  # Save the updated probabilities to history

    def observe_position(self):
        """
        Randomly select a position based on the current probabilities.

        Returns:
            int: Index of the selected position.
        """
        return np.random.choice(self.num_positions, p=self.probabilities)

    def get_probabilities(self):
        """
        Get the current probabilities.

        Returns:
            numpy.ndarray: Current probabilities.
        """
        return self.probabilities

    def plot_probabilities(self):
        """
        Plot the evolution of probabilities over time.
        Each line represents the probability of a specific position.
        """
        plt.figure(figsize=(10, 6))
        for i in range(self.num_positions):
            plt.plot(range(len(self.history)), [state[i] for state in self.history], label=f'Position {i}')
        plt.xlabel('Step')
        plt.ylabel('Probability')
        plt.title('Probability Evolution Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage
num_positions = 5  # Define the number of positions
quantum_state = QuantumState(num_positions)  # Create a QuantumState instance
for _ in range(10):
 quantum_state.update_probabilities(np.random.randint(0, 2))  # Update probabilities with random actions
quantum_state.plot_probabilities()  # Plot the probability evolution
