import torch
import torch.nn as nn
from typing import Tuple
from dynamic_env import DynamicEnv


# Constants for readability and reuse
DEFAULT_PRIORITY = 1.0
NUM_SAMPLES = 1
TENSOR_DTYPE = torch.float32  # Introduced for consistent tensor data type
INFO_KEY_DEFAULT = 0.0  # Default value for 'info' tensor if not used


class ConditionalActionsSystem(nn.Module):
    """Manages conditional actions and their priorities."""
    def __init__(self) -> None:
        super().__init__()
        self.conditions = {}
        self.priorities = {}
        self.actions = {}

    def add_action(self, *, condition: str, action, priority: float = DEFAULT_PRIORITY) -> None:
        """Adds a conditional action with a priority."""
        self.conditions[condition] = action
        self.priorities[condition] = priority


class Action:
    """Encapsulates an action to be performed in the environment."""
    def __init__(self, environment: DynamicEnv, conditional_actions_system: ConditionalActionsSystem, policy) -> None:
        self.environment = environment
        self.conditional_actions_system = conditional_actions_system
        self.policy = policy
        self.reward_log = []  # Renamed from reward_log for clarity

    def use_policy_to_select_action(self, action_policy: nn.Module, state: torch.Tensor) -> int:
        """Selects an action from the policy based on the current state."""
        return self._select_action_from_policy(action_policy, state)

    def execute_environment_action(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Executes action in the environment and returns new state, reward, and completion status."""
        new_state, reward, done = self.environment.execute_action(action)
        self._record_reward(reward)
        return new_state, reward, done

    def perform_policy_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Selects action based on the policy and then executes action in the environment."""
        action = self.use_policy_to_select_action(self.policy, state)
        return self.execute_environment_action(action)

    def _record_reward(self, reward: float) -> None:
        """Appends the given reward to the reward log."""
        self.reward_log.append(reward)
        # Log reward directly as there's no `store_experience` method
        self.reward_log.append(reward)

    @staticmethod
    def _select_action_from_policy(action_policy: nn.Module, state: torch.Tensor) -> int:
        """Selects action based on the policy and current state"""
        with torch.no_grad():
            probabilities, _ = action_policy(state)
        return torch.multinomial(probabilities, NUM_SAMPLES).item()