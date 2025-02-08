import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import pickle

# ------------------------------
# Placeholder Implementations
# ------------------------------


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


class CartPoleEnvironment:
    """Placeholder for the CartPoleEnvironment class."""
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

# ------------------------------
# CommandProcessor Class
# ------------------------------

class CommandProcessor:
    TERMINATION_THRESHOLD_RATIO = 0.5  # Threshold ratio for termination

    DEFAULT_COMMAND = "UNDEFINED_COMMAND"
    DEFAULT_FEEDBACK = "No feedback provided"

    def __init__(self):
        self.optimizer = None
        self.environment = self._initialize_environment()
        self.bayes_logic = BayesLogic()
        self.command_text = self.DEFAULT_COMMAND
        self.feedback_text = self.DEFAULT_FEEDBACK
        self._initialize_policy()
        self.time_interpreter = self._initialize_time_interpreter()

        # Initialize agent-related properties
        self.agent_trajectory = []
        self.log_probabilities = []
        self.value_estimates = []

        # Neural network components
        self._initialize_neural_network_components()

    @staticmethod
    def _initialize_environment():
        """Initializes and returns the CartPole environment."""
        try:
            # Attempt to import CartPoleEnvironment from external module
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
        """Initializes policy logic (placeholder)."""
        class Policy:
            def enforce(self):
                pass  # Placeholder for actual policy logic
        return Policy()

    def _initialize_neural_network_components(self):
        """Initializes neural network components."""
        self.actor_network = self._build_network(input_size=1, hidden_size=[10], output_size=2)
        self.critic_network = self._build_network(input_size=1, hidden_size=[10], output_size=1)
        self.optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def _build_network(self, input_size, hidden_size, output_size):
        """Builds a simple feedforward neural network."""
        layers = []
        previous_size = input_size
        for hidden in hidden_size:
            layers.append(torch.nn.Linear(previous_size, hidden))
            layers.append(torch.nn.ReLU())
            previous_size = hidden
        layers.append(torch.nn.Linear(previous_size, output_size))
        return torch.nn.Sequential(*layers)

    def initialize_time_interpreter(self, input_size, hidden_size, output_size, learning_rate):
        """Initializes the time interpreter and neural network training components."""
        self.time_interpreter = self._build_network(input_size, hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.time_interpreter.parameters(), lr=learning_rate)

    def set_feedback_text(self, feedback_text):
        """Sets the feedback text used for GUI interaction."""
        self.feedback_text = feedback_text

    def _is_termination_condition_met(self):
        """Checks if termination condition is met based on the environment's state."""
        return abs(
            self.environment.current_position - self.environment.target_position
        ) < self.environment.step_size * self.TERMINATION_THRESHOLD_RATIO

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
            case _:
                return "Action: Unknown"

    def get_initial_state(self) -> torch.Tensor:
        """Returns the initial state of the environment as a tensor."""
        return torch.tensor([self.environment.current_position], dtype=torch.float32)

    def set_state(self, state: torch.Tensor) -> None:
        """Sets the environment's state from the provided tensor state."""
        self.environment.current_position = state.item()
        if hasattr(self.environment, "get_state"):
            self.agent_trajectory.append(self.environment.get_state())

    def _calculate_cosines(self, state, env_value):
        """Uses the evaluator to calculate cosine values."""
        return self.environment.calculate_cosines_with_states(state, env_value)

    def _calculate_entropy_and_log(self, state):
        """Calculates entropy and logs the result."""
        entropy = self.bayes_logic.calculate_entropy(state)
        self.log_probabilities.append(entropy)
        print(f"Entropy calculated: {entropy}")
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
        print(f"Training loss: {loss.item()}")

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
            # Placeholder for actual interaction data and prediction
            interaction_data = {"result": np.random.randint(0, 2)}
            predicted_state = np.random.random()

            action = interaction_data["result"]  # Placeholder for action selection logic
            self._handle_feedback(action, interaction_data["result"])
            self._store_experience(termination_condition_met, action, interaction_data["result"])
        except Exception as error:
            self._handle_error(f"Error processing command: {error}")

    def _store_experience(self, termination, action, reward):
        """Stores the interaction experience."""
        # Placeholder for experience storage logic
        pass

    def log(self, message):
        """Logs a message (placeholder)."""
        print(message)

    def _handle_error(self, message):
        """Handles errors by logging and potentially displaying."""
        print(message)
        messagebox.showerror("Error", message)

# ------------------------------
# GUI Application
# ------------------------------

def _is_window_active(window: tk.Toplevel) -> bool:
    """Checks if the given window is active (exists and is open)."""
    return window is not None and window.winfo_exists()

class MultiWindowApp:
    """
    Provides the main interface to display and interact with the system.
    It is in charge of the UI and the calls to the processing logic.
    """

    # Constants
    MAIN_WINDOW_TITLE = "Panel de Control - Figuras 3D"
    MAIN_WINDOW_GEOMETRY = "500x400"
    GRAPH_WINDOW_TITLE = "Ventana de Gráfica 3D"
    GRAPH_WINDOW_GEOMETRY = "800x600"
    RESULTS_WINDOW_TITLE = "Ventana de Resultados"
    RESULTS_WINDOW_GEOMETRY = "400x300"
    DEFAULT_COMMAND_TEXT = "Escribe aquí..."
    DEFAULT_NO_DATA_TEXT = "No data sent yet."

    def __init__(self) -> None:
        """Initializes the main application window and its components."""
        # Initialize Command Processor
        self.command_processor = CommandProcessor()

        # Initialize Main Window
        self._initialize_main_window()

        self.shared_data = self.DEFAULT_NO_DATA_TEXT
        self.results_window = None
        self.graph_window = None
        self.result_label = None

    def _initialize_main_window(self) -> None:
        """Initializes the main window and UI elements."""
        self.main_window = tk.Tk()
        self.main_window.title(self.MAIN_WINDOW_TITLE)
        self.main_window.geometry(self.MAIN_WINDOW_GEOMETRY)

        # Create buttons and input elements
        self._create_command_button("Abrir Ventana de Resultados", self.open_results_window)
        self._create_command_button("Abrir Ventana de Gráficas", self.open_graph_window)

        # Shape Selection
        self.shape_var = tk.StringVar(value="Esfera")
        ttk.Label(self.main_window, text="Selecciona una figura geométrica:").pack(pady=5)
        shape_menu = ttk.OptionMenu(
            self.main_window,
            self.shape_var,
            "Esfera",
            "Esfera",
            "Cubo",
            "Pirámide",
            command=lambda _: self._update_parameter_fields()
        )
        shape_menu.pack(pady=10)

        # Parameter Inputs Frame
        self.parameter_frame = ttk.Frame(self.main_window)
        self.parameter_frame.pack(pady=10)
        self._update_parameter_fields()  # Initialize parameter fields based on default shape

        # Process Command Button
        self._create_command_button("Procesar Comando", self._process_command)

        # Additional Interaction Button (Optional)
        # You can add more buttons or inputs as needed

    def _create_command_button(self, text, command):
        """Helper method to create reusable buttons."""
        ttk.Button(self.main_window, text=text, command=command).pack(pady=10)

    def open_graph_window(self) -> None:
        """Opens the graph window and plots the selected shape."""
        if not _is_window_active(self.graph_window):
            self.graph_window = self._create_window(
                self.GRAPH_WINDOW_TITLE, self.GRAPH_WINDOW_GEOMETRY
            )
            selected_shape = self.shape_var.get()
            parameters = {}
            try:
                if selected_shape == "Esfera":
                    parameters["radio"] = float(self.radius_entry.get())
                elif selected_shape == "Cubo":
                    parameters["lado"] = float(self.side_entry.get())
                elif selected_shape == "Pirámide":
                    parameters["base"] = float(self.base_entry.get())
                    parameters["altura"] = float(self.height_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Por favor ingresa valores válidos.")
                return

            self._plot_3d_figure(selected_shape, parameters, self.graph_window)

    def open_results_window(self) -> None:
        """Opens the results window to display shared data."""
        if not _is_window_active(self.results_window):
            self.results_window = self._create_window(
                self.RESULTS_WINDOW_TITLE, self.RESULTS_WINDOW_GEOMETRY
            )
            self.result_label = ttk.Label(
                self.results_window, text=self.shared_data, wraplength=380
            )
            self.result_label.pack(pady=20)

    def send_data_to_results(self) -> None:
        """Sends data to the results window."""
        if not _is_window_active(self.results_window):
            messagebox.showerror("Error", "Primero abre la ventana de resultados.")
            return

        data = self._get_validated_command()
        if data:
            self.shared_data = f"Datos Enviados: {data}"
            self.result_label.config(text=self.shared_data)
            # Optionally, integrate with CommandProcessor here
            self.command_processor.set_feedback_text(self.result_label)
            self.command_processor.process_interaction()

    def _process_command(self) -> None:
        """Processes the command based on the selected shape."""
        command = self._get_validated_command()
        if command:
            messagebox.showinfo("Información", f"Figura seleccionada: {command}")
            self.send_data_to_results()

    def _get_validated_command(self) -> str:
        """Validates and retrieves the selected shape."""
        data = self.shape_var.get().strip()
        if not data:
            messagebox.showerror("Error", "No se seleccionó una figura.")
            return None
        return data

    def _create_window(self, title: str, geometry: str) -> tk.Toplevel:
        """Creates and returns a new Toplevel window."""
        new_window = tk.Toplevel(self.main_window)
        new_window.title(title)
        new_window.geometry(geometry)
        return new_window

    def _update_parameter_fields(self) -> None:
        """Update input fields dynamically based on the selected 3D shape."""
        for widget in self.parameter_frame.winfo_children():
            widget.destroy()

        selected_shape = self.shape_var.get()
        if selected_shape == "Esfera":
            ttk.Label(self.parameter_frame, text="Radio:").grid(row=0, column=0, padx=5, pady=5)
            self.radius_entry = ttk.Entry(self.parameter_frame, width=10)
            self.radius_entry.grid(row=0, column=1, padx=5, pady=5)
            self.radius_entry.insert(0, "1.0")  # Default value
        elif selected_shape == "Cubo":
            ttk.Label(self.parameter_frame, text="Longitud del Lado:").grid(row=0, column=0, padx=5, pady=5)
            self.side_entry = ttk.Entry(self.parameter_frame, width=10)
            self.side_entry.grid(row=0, column=1, padx=5, pady=5)
            self.side_entry.insert(0, "1.0")  # Default value
        elif selected_shape == "Pirámide":
            ttk.Label(self.parameter_frame, text="Longitud de la Base:").grid(row=0, column=0, padx=5, pady=5)
            self.base_entry = ttk.Entry(self.parameter_frame, width=10)
            self.base_entry.grid(row=0, column=1, padx=5, pady=5)
            self.base_entry.insert(0, "1.0")  # Default value

            ttk.Label(self.parameter_frame, text="Altura:").grid(row=1, column=0, padx=5, pady=5)
            self.height_entry = ttk.Entry(self.parameter_frame, width=10)
            self.height_entry.grid(row=1, column=1, padx=5, pady=5)
            self.height_entry.insert(0, "1.0")  # Default value

    def _plot_3d_figure(self, shape: str, params: dict, window: tk.Toplevel) -> None:
        """Generates and plots the selected 3D figure based on user parameters."""
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        if shape == "Esfera":
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            r = float(params["radio"])
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color="b", alpha=0.7)

        elif shape == "Cubo":
            lado = float(params["lado"])
            r = np.linspace(-lado / 2, lado / 2, 2)
            X, Y = np.meshgrid(r, r)
            ax.plot_surface(X, Y, -lado / 2, color="g", alpha=0.7)
            ax.plot_surface(X, Y, lado / 2, color="g", alpha=0.7)
            ax.plot_surface(X, -lado / 2, Y, color="g", alpha=0.7)
            ax.plot_surface(X, lado / 2, Y, color="g", alpha=0.7)
            ax.plot_surface(-lado / 2, X, Y, color="g", alpha=0.7)
            ax.plot_surface(lado / 2, X, Y, color="g", alpha=0.7)

        elif shape == "Pirámide":
            base = float(params["base"])
            altura = float(params["altura"])
            # Define the base square
            vertices = np.array([
                [0, 0, altura],
                [-base / 2, -base / 2, 0],
                [base / 2, -base / 2, 0],
                [base / 2, base / 2, 0],
                [-base / 2, base / 2, 0]
            ])
            # Plot base
            ax.plot_trisurf(vertices[1:, 0], vertices[1:, 1], vertices[1:, 2], color="r", alpha=0.7)
            # Plot sides
            for i in range(1, len(vertices)):
                j = i + 1 if i + 1 < len(vertices) else 1
                x = [vertices[0, 0], vertices[i, 0], vertices[j, 0], vertices[0, 0]]
                y = [vertices[0, 1], vertices[i, 1], vertices[j, 1], vertices[0, 1]]
                z = [vertices[0, 2], vertices[i, 2], vertices[j, 2], vertices[0, 2]]
                ax.plot_trisurf(x, y, z, color="r", alpha=0.7)

        ax.set_title(f"Figura 3D: {shape}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run(self) -> None:
        """Executes the main loop of the application."""
        self.main_window.mainloop()

# ------------------------------
# Main Execution
# ------------------------------

def create_app() -> MultiWindowApp:
    """Creates and returns an instance of the MultiWindowApp."""
    return MultiWindowApp()

