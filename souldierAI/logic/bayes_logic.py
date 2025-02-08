from souldierAI.neuralQ.neural_quantum import NeuralNetwork


class BayesLogic:
    """
    Implements Bayesian logic for updating beliefs based on state, actions and external influences.
    It provides methods to calculate probabilities, priors and related inferences.
    """

    def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
        """
        Calculates the posterior probability using Bayes' theorem.
        """
        EPSILON = 1e-6
        return (conditional_b_given_a * prior_a) / (prior_b or EPSILON)

    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calculates a conditional probability given a joint probability and a prior.
        """
        EPSILON = 1e-6
        prior = prior if prior != 0 else EPSILON
        return joint_probability / (prior or EPSILON)

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Get prior based on entropy value.
        """
        return 0.3 if entropy > 0.8 else 0.1

    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Get prior based on coherence value.
        """
        return 0.6 if coherence > 0.6 else 0.2

    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        """
        Calculates the joint probability of A and B based on coherence and action.
        """
        return (prn_influence * 0.8 + (1 - prn_influence) * 0.2) if coherence > 0.6 and action == 1 else (prn_influence * 0.1 + (1 - prn_influence) * 0.7) if coherence > 0.6 else 0.3

    def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, prn_influence: float,
                                                  action: int) -> dict:
        """
        Calculates probabilities and selects an action based on entropy, coherence,
        PRN influence, and the action.
        """
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        # Calculations based on influence
        posterior_a_given_b = self.calculate_posterior_probability(high_entropy_prior, high_coherence_prior, (prn_influence * 0.7 + (1 - prn_influence) * 0.3) if entropy > 0.8 else 0.2)

        joint_probability_ab = self.calculate_joint_probability(coherence, action, prn_influence)
        conditional_action_given_b = self.calculate_conditional_probability(joint_probability_ab, high_coherence_prior)

        # Action selection logic
        action_to_take = 1 if conditional_action_given_b > 0.5 else 0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b,
        }

import torch
import torch.optim as optim

from dynamic_env import DynamicEnv
from evaluator import Evaluator
from state import State

# Constants for readability and reuse
DEFAULT_PRIORITY = 1.0
HIDDEN_DIM = 128
DISCOUNT_FACTOR = 0.99
STATE_DIM = 4
ACTION_DIM = 2
LEARNING_RATE = 0.01
TENSOR_DTYPE = torch.float32  # Introduced for consistent tensor data type
INFO_KEY_DEFAULT = 0.0  # Default value for 'info' tensor if not used


def _initialize_policy():
    """Initializes the policy logic."""

    class Policy:
        def enforce(self):
            pass  # Placeholder for policy logic

    return Policy()


def _select_action_from_cosenos(cosenos, prn_results):
    """Selecciona la acción a partir de los cosenos y los resultados de las PRN"""
    # Implementa aquí la lógica para seleccionar la acción a partir de los cosenos
    if cosenos[0] > 0.5:
        return 1
    elif cosenos[0] < -0.5:
        return 0
    else:
        return 2


def _calculate_cosenos_directores(vector):
    """Calcula los cosenos directores"""
    # Implementa aquí la lógica para calcular los cosenos directores
    norm = torch.linalg.norm(vector)
    return vector / norm if norm > 0 else torch.tensor([0, 0, 0],
                                                       dtype=torch.float32)  # Reemplaza esto con la logica correcta

class CommandProcessor:
    """
    Orchestrates the entire interaction process of the agent.
    This class is in charge of all the interactions and logic of the agent.
    """
    DEFAULT_COMMAND = "UNDEFINED_COMMAND"
    DEFAULT_ENV_VALUE = 1
    DEFAULT_FEEDBACK = "No feedback provided"
    ACTIVATION_FUNCTION = "relu"
    LEARNING_RATE = 0.001  # Example constant for learning rate
    """
    Orchestrates the entire interaction process of the agent.
    This class is in charge of all the interactions and logic of the agent.
    """
    def __init__(self) -> None:
        """Initializes the command processor, agent components, and neural networks."""
        self._initialize_environment()
        self.evaluator = Evaluator()
        self.bayes_logic = BayesLogic()

        # Command and feedback initialization
        self.command_text = self.DEFAULT_COMMAND
        self.feedback_text = self.DEFAULT_FEEDBACK

        # Policy and interpreter initialization
        self.policy = self._initialize_policy()
        self.time_interpreter = None

        # Initialize agent-related and neural network properties
        self._initialize_agent_properties()
        self._initialize_network_components()


    def _initialize_environment(self) -> None:
        """Initializes the environment and its starting state."""
        initial_state = State(position=0, velocity=0, angle=0, angular_velocity=0)
        self.environment = DynamicEnv(initial_state=initial_state)
        self.environment = DynamicEnv(initial_state=initial_state)
        self.evaluator = Evaluator()
        self.bayes_logic = BayesLogic()

        # Command and feedback initialization
        self.command_text = self.DEFAULT_COMMAND
        self.feedback_text = self.DEFAULT_FEEDBACK

        # Policy and interpreter initialization
        self.policy = self._initialize_policy()
        self.time_interpreter = None

        # Initialize agent-related and neural network properties
        self._initialize_agent_properties()
        self._initialize_network_components()

    def _initialize_agent_properties(self) -> None:
        """Initialize agent-related properties."""
        self.agent_trajectory = []
        self.log_probabilities = []
        self.value_estimates = []

    def _initialize_network_components(self) -> None:
        """Initializes all components related to neural networks."""
        self.actor_network = None
        self.critic_network = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.criterion = None
        self.prn_manager = None  # Placeholder for PRN manager
        self.quantum_neuron = None
        self.init_networks(state_size=STATE_DIM, action_size=ACTION_DIM)

    def init_networks(self, state_size: int, action_size: int) -> None:
        """Initializes actor and critic networks with optimizers."""
        self._initialize_actor_network(state_size, action_size)
        self._initialize_critic_network(state_size=state_size)

    def _initialize_actor_network(self, state_size: int, action_size: int) -> None:
        """Initializes the actor network and its optimizer."""
        self.actor_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=[HIDDEN_DIM, HIDDEN_DIM],  # Fixed hidden layer size
            output_size=action_size,
            activation=self.ACTIVATION_FUNCTION  # Fixed activation function
        )
        self.optimizer_actor = optim.Adam(
        self.actor_network.activate_derivative(self.actor_network.weights[0]),
        lr=self.LEARNING_RATE  # Fixed learning rate reference
        )

    def _initialize_critic_network(self):
        self.bayes_logic = BayesLogic()
