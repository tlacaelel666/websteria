import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from torch.nn import Sequential as NeuralNetwork

# Constants for readability and reuse
DEFAULT_PRIORITY = 1.0
HIDDEN_DIM = 128
DISCOUNT_FACTOR = 0.99
STATE_DIM = 4
ACTION_DIM = 2
LEARNING_RATE = 0.01
TENSOR_DTYPE = torch.float32  # Introduced for consistent tensor data type
INFO_KEY_DEFAULT = 0.0  # Default value for 'info' tensor if not used - TODO: Ensure its correct usage.


def _initialize_policy():
    """Initializes the policy logic."""

    class Policy:
        def enforce(self):
            pass  # Placeholder for policy logic

    return Policy()


class CommandProcessor:
    # Constants for initialization
    DEFAULT_COMMAND = "UNDEFINED_COMMAND"
    DEFAULT_ENV_VALUE = 1
    DEFAULT_FEEDBACK = "No feedback provided"
    ACTIVATION_FUNCTION = "relu"
    LEARNING_RATE = 0.001  # Example constant for learning rate

    def __init__(self):
        # Initialize environment and external modules
        initial_state = State(position=0, velocity=0, angle=0, angular_velocity=0)
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

    def _initialize_agent_properties(self):
        """Initialize agent-related properties."""
        self.agent_trajectory = []
        self.log_probabilities = []
        self.value_estimates = []

    def _initialize_network_components(self):
        """Initialize components related to neural networks."""
        self.actor_network = None
        self.critic_network = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.criterion = None
        self.prn_manager = None  # Placeholder for PRN manager
        self.quantum_neuron = None
        self.init_networks(state_size=STATE_DIM, action_size=ACTION_DIM)

    def init_networks(self, state_size, action_size):
        """Initialize actor and critic networks with optimizers."""
        self._initialize_actor_network(state_size, action_size)
        self._initialize_critic_network(state_size=state_size)

    def _initialize_network(self, _initialize_policy):
        self.actor_network = NeuralNetwork()
        self.critic_network = NeuralNetwork()
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def _initialize_policy(self):
        pass


def _initialize_actor_network(self, state_size, action_size):
    """Initialize the actor network and its optimizer."""
    
    self.actor_network = NeuralNetwork(
        nn.Linear(state_size, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, action_size)
    )
    self.optimizer_actor = optim.Adam(
    self.actor_network.parameters(),
    lr=LEARNING_RATE  # Fixed learning rate reference
        )


    def _initialize_critic_network(self, state_size):
        """Initialize the critic network and its optimizer."""
        self.critic_network = NeuralNetwork(
            input_size=state_size,
            hidden_size=[HIDDEN_DIM, HIDDEN_DIM],  # Fixed hidden layer size
            output_size=1,
            activation="relu"  # Fixed activation function
        )
        self.optimizer_critic = optim.Adam(
            self.critic_network.parameters(),
            lr=LEARNING_RATE  # Fixed learning rate reference
        )

    def _initialize_policy(self):
        """Initializes and returns the policy."""
        return _initialize_policy()  # Placeholder for actual policy initialization
from evaluator import Evaluator
from state import State
from PRN.time_interpretator import TimeInterpretator  # Ensure the 'time_interpretator' module exists and is in your PYTHONPATH, or create it if missing
from dynamic_env import DynamicEnv
from bayes_logic import BayesLogic

# Constants for readability and reuse
def _initialize_policy():
    """Initializes the policy logic."""

    class Policy:
        DEFAULT_PRIORITY = 1.0
        
        def enforce(self):
            pass  # Placeholder for policy logic

    return Policy()
HIDDEN_LAYER_DIM = 128
class CommandProcessor:
    DISCOUNT_FACTOR = 0.99
STATE_DIM = 4
ACTION_DIM = 2
LEARNING_RATE = 0.01
class CommandProcessor:
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
    DEFAULT_COMMAND = "UNDEFINED_COMMAND"
    DEFAULT_ENV_VALUE = 1
    DEFAULT_FEEDBACK = "No feedback provided"

    def __init__(self):
        initial_state = State(position=0, velocity=0, angle=0, angular_velocity=0)
        self.environment = DynamicEnv(initial_state=initial_state)
        self.modulo_evaluacion = Evaluator()
        self.bayes_logic = BayesLogic()
        self.command_text = CommandProcessor.DEFAULT_COMMAND
        self.feedback_text = CommandProcessor.DEFAULT_FEEDBACK
        self.policy = self._initialize_policy()
        self.time_interpretator = None

        # Initialize agent-related properties
        self.agent_trajectory = []
        self.log_probabilities = []
        self.value_estimates = []

        # Initialize neural network components
        self.actor_network = None
        self.critic_network = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.criterion = None

        # Initialize QuantumNeuron
        self.prn_manager = None  # Placeholder for PRN manager
        self.quantum_neuron = None
        self.init_networks(state_size=STATE_DIM, action_size=ACTION_DIM, hidden_size=[HIDDEN_DIM, HIDDEN_DIM])
    def init_networks(self, state_size, action_size, hidden_size):
        self.actor_network = NeuralNetwork(input_size=state_size, hidden_size=hidden_size, output_size=action_size, activation="relu")
        self.critic_network = NeuralNetwork(input_size=state_size, hidden_size=hidden_size, output_size=1, activation="relu")
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)  # Ajusta la tasa de aprendizaje
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)  # Ajusta la tasa de aprendizaje

    def _initialize_policy(self):
        """Initializes and returns the policy."""
        return _initialize_policy()

    def initialize_time_interpreter(self, input_size, hidden_size, output_size, learning_rate):
        """Initializes the time interpretator and its requirements."""
        self.time_interpretator = TimeInterpretator(input_size, hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.time_interpretator.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def set_feedback_text(self, feedback_text):
        """Sets the feedback text used for GUI interaction."""
        self.feedback_text = feedback_text

    def _get_previous_state(self):
        """Fetches the current state and checks termination condition."""
        return self.environment.get_state()

    def get_state_text(self) -> str:
        """Returns a string representing the current state of the environment"""
        return self.environment.render()

    def get_action_text(self, action: int) -> str:
        """Returns a string representing the action."""
        if action == 0:
            return "Action: Move Left"
        elif action == 1:
            return "Action: Move Right"

    def get_initial_state(self) -> torch.Tensor:
        """Returns the initial state of the environment as a tensor."""
        return self.environment.get_initial_state()

    def set_state(self, state: torch.Tensor) -> None:
        """Set the state of the environment from a given state."""
        self.environment.set_state(state)
        self.agent_trajectory.append(self.environment.get_state())

    def _calculate_cosines(self, state, env_value):
        """Uses the evaluator to calculate cosine values."""
        return self.modulo_evaluacion.calculate_cosines_with_states(state, env_value)

    def _calculate_entropy_and_log(self, state):
        """Calculates entropy and logs the result."""
        entropy = self.modulo_evaluacion.calculate_entropy(state)
        self.log(f"Entropy calculated: {entropy}")
        return entropy

    def _update_and_train_interpreter(self, state, action):
        """Updates the time interpreter with state and action."""
        x = torch.tensor([[state[0]]], dtype=torch.float32)  # Using position for training
        y = torch.tensor([action], dtype=torch.float32)
        self.optimizer.zero_grad()
        output = self.time_interpretator(x)
        loss = self.criterion(output, y.long())
        loss.backward()
        self.optimizer.step()

    def _handle_feedback(self, state, action, reward):
        """Handles feedback display and anomaly detection."""
        if self.feedback_text is not None:
            state_text = self.environment.render()
            feedback = f"Command: {self.command_text} -> Action: {action}. {state_text}. Reward: {reward}\n"
            self.feedback_text.config(text=feedback)


    def process_interaction(self):
        try:
            previous_state = self._get_previous_state()
            interaction_data, predicted_state = self._evaluate_and_predict(previous_state)
            action = self._select_and_execute_action(previous_state, interaction_data)
            interaction_data["future_state"] = predicted_state
            self._handle_feedback(previous_state, action, interaction_data["result"])
            self._store_experience(previous_state, action, interaction_data["result"])
        except Exception as error:
            self._handle_error(f"Error processing command: {error}")

    def _evaluate_and_predict(self, state_previous):
        """
        Evaluates the quality of the interaction and predicts the future state.
        """
        interaction_data = {
            "state": state_previous,
            "action": None,
            "result": None,
            "command": self.command_text,
        }
        # Evaluate quality and predict future state (Placeholder)
        interaction_data["result"] = 1

        # Predict future state using cosines
        env_value = 1  # Assume this method provides a relevant environment value
        cos_x, cos_y, cos_z = self.modulo_evaluacion.calculate_cosines_with_states(state_previous, env_value)
        interaction_data["cosines"] = (cos_x, cos_y, cos_z)
        future_state = self._predict_next_state(interaction_data)
        return interaction_data, future_state

    def _predict_next_state(self, interaction_data):
        """Predicts the next state using the interaction data (Placeholder)"""
        return interaction_data["state"]  # Placeholder: No actual prediction

    def _select_and_execute_action(self, state_previous, interaction_data):
        """
        Uses BayesLogic to calculate probabilities and select the optimal action.
        """
        # Retrieve coherence and entropy
        object_state = state_previous
        entropy = self.modulo_evaluacion.calculate_entropy(object_state)
        coherence = self.modulo_evaluacion.evaluate_coherence(object_state)

        # Use BayesLogic for action selection
        prn_influence = 0.5  # Placeholder
        #action_probs, _ = self.actor_network(state_previous)
        probabilities = self.bayes_logic.calculate_probabilities_and_select_action(entropy, coherence, prn_influence, 0)
        proposed_action = self._select_action(state_previous)  # Use the function with the cosines and actor critic
        # Execute action and update interaction data
        new_state, reward, _ = self.environment.execute_action(proposed_action)
        interaction_data["action"] = proposed_action
        interaction_data["result"] = reward

        # Update time interpretator
        if self.time_interpretator is not None:
            self._update_and_train_interpreter(state_previous, proposed_action)

        return proposed_action

    def _store_experience(self, state_previous, proposed_action, reward):
        """
        Saves the interaction data for future use.
        """
        # Placeholder
        pass

    def log(self, message):
        """
        Utility method for logging information.
        """
        print(f"[LOG] {message}")
    def _select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Adds batch dimension and specifies dtype
        action_probs, state_value = self.actor_network(state)
        m = dist.Categorical(action_probs)  # Uses a categorical distribution to sample the action
        action = m.sample()
        log_prob = m.log_prob(action)
        self.log_probabilities.append(log_prob)
        self.value_estimates.append(state_value)
        return action.item()  # Returns an integer

    def training_models(self, discounted_rewards):
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        values = torch.stack(self.value_estimates)
        advantage = discounted_rewards - values
        critic_loss = advantage.pow(2).mean()
        log_probs = torch.stack(self.log_probabilities)
        actor_loss = (-log_probs * advantage.detach()).mean()  # detach() para la ventaja

        # -----> Update networks <-----
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        self.optimizer_critic.zero_grad()  # Ensure that the critic's optimizer is defined
        critic_loss.backward()
        self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()

    def calculate_return(self, rewards, gamma):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns

    def plot_trajectory_3d(self):
        """Genera un gráfico 3D de la trayectoria del agente, usando las tres primeras dimensiones del estado."""
        if not self.agent_trajectory:
            print("No trajectory data available for plotting.")
            return
            
            # Extract the first three dimensions of the trajectory
        x = [s[0].item() for s in self.agent_trajectory]  # .item() para obtener el valor del tensor
        y = [s[1].item() for s in self.agent_trajectory]
        z = [s[2].item() for s in self.agent_trajectory]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Trayectoria del agente')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def _apply_prn(self, state):
        """
        Aplica las políticas, reglas, y normas al estado actual
        y regresa los resultados, como por ejemplo, rangos de cosenos directores y acciones asociadas
        """
        # Implementa aquí la lógica de la PRN_factory o la lógica de las PRN.
        # Ejemplo:
        # 1. Ordenar los estados usando las PRN
        # ordered_states = self.prn_factory.order_states(states)
        # 2. Mapear los estados a un conjunto de cosenos directores
        # mapped_states = self.prn_factory.map_state_to_action(ordered_states)
        return {}  # Reemplaza {} por la información resultante

    def _apply_bayes(self, state, previous_action):
        """
        Aplica la lógica bayesiana para actualizar las creencias sobre el estado actual
        y regresar la información obtenida de la acción previa
        """
        # Implementa aquí la lógica bayesiana
        # 1. Calcular la probabilidad a priori
        # 2. Calcular la verosimilitud
        # 3. Calcular la probabilidad a posteriori
        # 4. Guardar la información resultante.
        return {}  # Reemplaza {} por la información resultante

    def _arco_bayes_function(self, s, prn_results, H, C, B):
        """
        Aplica la lógica arCo Bayes para determinar la acción
        """
        # 1. Obtener la información del estado
        # 2. Obtener la información de las PRN
        # 3. Obtener la entropía
        # 4. Obtener la coherencia
        # 5. Obtener la información bayesiana del paso anterior
        # 6. Seleccionar la acción utilizando la lógica de los cosenos directores
        # y la información obtenida anteriormente.

        # Ejemplo:
        # 1. Mapear el estado a un vector 3D
        vector = self._map_state_to_vector(s)
        # 2. Calcular los cosenos directores del vector
        cosenos = _calculate_cosenos_directores(vector)
        # 3. Seleccionar la acción que corresponde al rango de ángulos
        action = _select_action_from_cosenos(cosenos, prn_results)

        return action  # Reemplaza 1 con la acción seleccionada

    # Métodos Auxiliares (Puedes definirlos en clases auxiliares)
    def _map_state_to_vector(self, state):
        """Mapea el estado a un vector 3D"""
        # Implementa aquí la lógica para mapear el estado a un vector 3D
        return torch.tensor([state[0], state[1], state[2]], dtype=torch.float32)  # Reemplaza esto con la logica correcta

