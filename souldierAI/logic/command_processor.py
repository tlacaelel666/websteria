import torch

from souldierAI.logic.bayes_logic import BayesLogic


TERMINATION_THRESHOLD_RATIO = 0.5  # Threshold ratio for termination


class CommandProcessor:
    DEFAULT_COMMAND = "UNDEFINED_COMMAND"
    DEFAULT_FEEDBACK = "No feedback provided"

    def __init__(self):
        self.optimizer = None
        self.environment = self._initialize_environment()
        self.bayes_logic = BayesLogic()
        self.command_text = self.DEFAULT_COMMAND
        self.feedback_text = self.DEFAULT_FEEDBACK
        self._initialize_policy()
        self.time_interpreter = None

        # Initialize agent-related properties
        self.agent_trajectory = []
        self.log_probabilities = []
        self.value_estimates = []

        # Neural network components
        self._initialize_neural_network_components()

    @staticmethod
    def _initialize_environment():
            """Initializes and returns the CartPole environment. Provides a fallback if unavailable."""
            try:
                from logic.dynamic_env import CartPoleEnvironment
                return CartPoleEnvironment()
            except ImportError:
                # Fallback implementation if CartPoleEnvironment cannot be imported
                class BasicEnvironment:
                    def __init__(self):
                        self.current_position = 0
                        self.target_position = 100
                        self.step_size = 1

                    def get_state(self):
                        return {"position": self.current_position, "target": self.target_position}

                    @staticmethod
                    def calculate_cosines_with_states(state, env_value):
                        return state * env_value

                    def render(self):
                        return f"Position: {self.current_position}, Target: {self.target_position}"

                return BasicEnvironment()
                
    @staticmethod
    def _initialize_policy():
        """
        Defines the policy logic.  This is a placeholder and needs a proper implementation.
        """
        class Policy:
            def enforce(self):
                pass  # Placeholder for policy logic

    def _initialize_neural_network_components(self):
        """Initializes neural network components."""
        self.actor_network = None
        self.critic_network = None
        self.optimizer = None
        self.criterion = None

    def initialize_time_interpreter(self, input_size, hidden_size, output_size, learning_rate):
        """Initializes the time interpreter and neural network training components."""
        self.time_interpreter(self)
        self.optimizer = torch.optim.Adam(self.time_interpreter.parameters(), lr=learning_rate)

    def set_feedback_text(self, feedback_text):
        """Sets the feedback text used for GUI interaction."""
        self.feedback_text = feedback_text

    def _is_termination_condition_met(self):
        """Checks if termination condition is met based on the environment's state."""
        return abs(
            self.environment.current_position - self.environment.target_position) < self.environment.step_size * TERMINATION_THRESHOLD_RATIO

    def get_state_text(self) -> str:
        """Returns a string representing the current state of the environment."""
        return f"Current Position: {self.environment.current_position}"

    @staticmethod
    def get_action_text(action: int) -> str:
        """Returns a string representing the action."""
        match action:
            case 0:
                return "Action: Move Left"
            case 1:
                return "Action: Move Right"
            case _:  # Default case
                return "Action: Unknown"


def get_initial_state(self) -> torch.Tensor:  #Corrected indentation
    """Returns the initial state of the environment as a tensor."""
    return torch.tensor([self.environment.current_position], dtype=torch.float32)


def set_state(self, state: torch.Tensor) -> None:  #Corrected indentation
    """Sets the environment's state from the provided tensor state."""
    self.environment.current_position = state.item()
    if hasattr(self.environment, "get_state"):
        self.agent_trajectory.append(self.environment.get_state())


def _calculate_cosines(self, state, env_value):  #Corrected indentation
    """Uses the evaluator to calculate cosine values."""
    return self.environment.calculate_cosines_with_states(state, env_value)


def _calculate_entropy_and_log(self, state):
    """Calculates entropy and logs the result."""
    entropy = self.bayes_logic.calculate_entropy(state)
    self.log(f"Entropy calculated: {entropy}")
    return entropy


def _update_and_train_interpreter(self, state, action):
    """Updates the time interpreter with state and action."""
    x = torch.tensor([[state[0]]], dtype=torch.float32)  # Using position for training
    y = torch.tensor([action], dtype=torch.float32)
    self.optimizer.zero_grad()
    output = self.time_interpreter(x)
    loss = self.criterion(output, y.long())
    loss.backward()
    self.optimizer.step()


def _handle_feedback(self, action, reward):
    """Handles feedback display and anomaly detection."""
    if self.feedback_text is not None:
        state_text = self.environment.render()
        feedback = f"Command: {self.command_text} -> Action: {action}. {state_text}. Reward: {reward}\n"
        self.feedback_text.config(text=feedback)


def process_interaction(self):
    """Processes a single interaction with the environment."""
    try:
        termination_condition_met = self._is_termination_condition_met()
        interaction_data, predicted_state = self._evaluate_and_predict(termination_condition_met)
        action = self._select_and_execute_action(termination_condition_met, interaction_data)
        interaction_data["future_state"] = predicted_state
        self._handle_feedback(action, interaction_data["result"])
        self._store_experience(termination_condition_met, action, interaction_data["result"])
    except Exception as error:
        self._handle_error(f"Error processing command: {error}")
