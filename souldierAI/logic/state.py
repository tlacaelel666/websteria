import random
from typing import Tuple
import torch

# Constants for position and motion limits
MAX_POSITION = 200
MIN_POSITION = -200
MAX_VELOCITY = 10
MAX_ANGLE = 30
MAX_ANGULAR_VELOCITY = 10


class State:
    """Encapsulates the position and motion state."""

    def __init__(self, position: float, velocity: float, angle: float, angular_velocity: float) -> None:
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.angular_velocity = angular_velocity


class DynamicEnv:
    """
    Manages the physics and state transitions for the dynamic environment.
    """

    def __init__(self, initial_state: State, step_size: float, time_step: float = 0.1) -> None:
        self.limits = (MIN_POSITION, MAX_POSITION)
        self.state = initial_state
        self.step_size = step_size
        self.time_step = time_step
        self.target_position = self._generate_target_position()
        self.terminated = False

    def _generate_target_position(self) -> float:
        """Randomly generates a target position within limits."""
        return random.uniform(self.limits[0], self.limits[1])

    def get_state(self) -> torch.Tensor:
        """Returns the current state as a normalized tensor."""
        return torch.tensor(self._normalize(), dtype=torch.float32)

    def _normalize(self) -> Tuple[float, float, float, float]:
        """Normalizes state attributes for consistency."""
        return (
            (self.state.position - self.limits[0]) / (self.limits[1] - self.limits[0]),
            self.state.velocity / MAX_VELOCITY,
            self.state.angle / MAX_ANGLE,
            self.state.angular_velocity / MAX_ANGULAR_VELOCITY
        )

    def execute_action(self, reaction: int) -> Tuple[torch.Tensor, float, bool]:
        """Executes an action and returns the next state, reward, and termination status."""
        if self.terminated:
            return self.get_state(), 0, self.terminated

        # Adjust velocity and angular velocity based on action
        adjustment = self.step_size if reaction == 1 else -self.step_size
        self.state.velocity += adjustment
        self.state.angular_velocity += adjustment

        # Update position and angle
        self.state.position += self.state.velocity
        self.state.angle += self.state.angular_velocity

        # Enforce boundaries
        self._enforce_boundaries()

        # Apply random perturbation
        if random.random() < 0.05:
            self.state.velocity += adjustment
            self.state.angular_velocity += adjustment

        # Calculate reward and check termination
        reward_state = self._calculate_reward()
        self.terminated = self._check_termination()
        return self.get_state(), reward_state, self.terminated

    def _enforce_boundaries(self) -> None:
        """Ensures that all state attributes remain within defined limits."""
        self.state.position = max(self.limits[0], min(self.state.position, self.limits[1]))
        self.state.velocity = max(-MAX_VELOCITY, min(self.state.velocity, MAX_VELOCITY))
        self.state.angle = max(-MAX_ANGLE, min(self.state.angle, MAX_ANGLE))
        self.state.angular_velocity = max(-MAX_ANGULAR_VELOCITY, min(self.state.angular_velocity, MAX_ANGULAR_VELOCITY))

    def _calculate_reward(self) -> float:
        """Calculates the reward based on the current position."""
        distance = abs(self.state.position - self.target_position)
        return 10 if distance < self.step_size else -distance / (self.limits[1] - self.limits[0])

    def _check_termination(self) -> bool:
        """Checks if the termination condition has been reached."""
        return abs(self.state.position - self.target_position) < self.step_size

    def render(self) -> str:
        """Returns a string representing the current state."""
        return (f"Position: {self.state.position:.2f}, Velocity: {self.state.velocity:.2f}, "
                f"Angle: {self.state.angle:.2f}, Angular Velocity: {self.state.angular_velocity:.2f}, "
                f"Target Position: {self.target_position:.2f}")

    def set_state(self, state_tensor: torch.Tensor) -> None:
        """Sets the state attributes from a normalized tensor."""
        self.state.position = state_tensor[0].item() * (self.limits[1] - self.limits[0]) + self.limits[0]
        self.state.velocity = state_tensor[1].item() * MAX_VELOCITY
        self.state.angle = state_tensor[2].item() * MAX_ANGLE
        self.state.angular_velocity = state_tensor[3].item() * MAX_ANGULAR_VELOCITY
