import numpy as np
from typing import Tuple

def shannon_entropy(data: list) -> float:
    """
    Calculates the Shannon entropy of a data set.

    Args:
      data (list or numpy.ndarray): List or array of data.

    Returns:
      float: Shannon entropy in bits.
    """
    # 1. Count occurrences of each unique value:
    values, counts = np.unique(data, return_counts=True)

    # 2. Calculate probabilities:
    probabilities = counts / len(data)

    # 3. Avoid logarithms of zero:
    probabilities = probabilities[probabilities > 0]

    # 4. Calculate the entropy:
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def calculate_cosines(entropy: float, env_value: float) -> Tuple[float, float, float]:
    """
    Calculates the directional cosines (x, y, z) for a three-dimensional vector.

    Args:
      entropy (float): Shannon entropy (x).
      env_value (float): Contextual environment value (y).

    Returns:
      tuple: Directional cosines (cos_x, cos_y, cos_z).
    """
    # Ensure to avoid division by zero:
    if entropy == 0:
        entropy = 1e-6
    if env_value == 0:
        env_value = 1e-6

    # Magnitude of the three-dimensional vector:
    magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)

    # Calculation of directional cosines:
    cos_x = entropy / magnitude
    cos_y = env_value / magnitude
    cos_z = 1 / magnitude

    return cos_x, cos_y, cos_z

# Example of usage:
if __name__ == "__main__":
    # Test data:
    sample_data = [1, 2, 3, 4, 5, 5, 2]
    entropy = shannon_entropy(sample_data)
    env_value = 0.8  # Example of an environment value

    cos_x, cos_y, cos_z = calculate_cosines(entropy, env_value)

    print(f"Entropy: {entropy:.4f}")
    print(f"Directional cosines: cos_x = {cos_x:.4f}, cos_y = {cos_y:.4f}, cos_z = {cos_z:.4f}")