import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib import pyplot as plt

from souldierAI.logic.command_processor import CommandProcessor


def _is_window_active(window):
    """Checks if the given window is active (exists and is open)."""
    return window is not None and window.winfo_exists()


class MultiWindowApp:
    # Constants
    MAIN_WINDOW_TITLE = "Panel de Control - Figuras 3D"
    MAIN_WINDOW_GEOMETRY = "400x300"
    GRAPH_WINDOW_TITLE = "Ventana de Gráfica 3D"
    GRAPH_WINDOW_GEOMETRY = "600x500"
    RESULTS_WINDOW_TITLE = "Ventana de Resultados"
    RESULTS_WINDOW_GEOMETRY = "400x300"
    DEFAULT_COMMAND_TEXT = "Escribe aquí..."
    DEFAULT_NO_DATA_TEXT = "Sin datos enviados aún"

    def __init__(self):
        # Initialize Main Window
        self._initialize_main_window()
        self.shared_data = self.DEFAULT_NO_DATA_TEXT
        self.results_window = None
        self.graph_window = None
        self.result_label = None
        self.command_processor = CommandProcessor()

    def _initialize_main_window(self):
        self.main_window = tk.Tk()
        self.main_window.title(self.MAIN_WINDOW_TITLE)
        self.main_window.geometry(self.MAIN_WINDOW_GEOMETRY)

        # Create buttons and input elements
        self._create_command_button("Abrir Ventana de Resultados", self.open_results_window)
        self._create_command_button("Abrir Ventana de Gráficas", self.open_graph_window)
        self.shape_var = tk.StringVar(value="Esfera")
        ttk.Label(self.main_window, text="Selecciona una figura geométrica:").pack(pady=5)
        shape_menu = ttk.OptionMenu(self.main_window, self.shape_var, "Esfera", "Esfera", "Cubo", "Pirámide")
        shape_menu.pack(pady=10)
        self.parameter_frame = ttk.Frame(self.main_window)
        self.parameter_frame.pack(pady=10)

        self.shape_var.trace("w", lambda *args: self._update_parameter_fields())

        # Process command button
        ttk.Button(
            self.main_window, text="Process Command", command=self._process_command
        ).pack(pady=10)

    def _create_command_button(self, text, command):
        """Helper method to create reusable buttons."""
        ttk.Button(self.main_window, text=text, command=command).pack(pady=10)

    def open_graph_window(self):
        """Opens the graph window and creates a 3D scatter plot."""
        if not _is_window_active(self.graph_window):
            self.graph_window = self._create_window(
                self.GRAPH_WINDOW_TITLE, self.GRAPH_WINDOW_GEOMETRY
            )
            x_data = np.linspace(-5, 5, 100)
            y_data = np.linspace(-5, 5, 100)
            z_data = np.sin(np.sqrt(x_data ** 2 + y_data ** 2))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x_data, z_data, label="Eje X")
            ax.plot(y_data, z_data, label="Eje Y")
            ax.set_title("Gráfica 3D")
            plt.show()

    def open_results_window(self):
        """Opens the results window to display shared data."""
        if not _is_window_active(self.results_window):
            self.results_window = self._create_window(
                self.RESULTS_WINDOW_TITLE, self.RESULTS_WINDOW_GEOMETRY
            )
            self.result_label = ttk.Label(self.results_window, text=self.shared_data, wraplength=300)
            self.result_label.pack(pady=20)

    def send_data_to_results(self):
        """Sends input data to the results window."""
        if not _is_window_active(self.results_window):
            messagebox.showerror("Error", "Primero abre la ventana de resultados.")
            return

        data = self._get_validated_command()
        self.shared_data = f"Datos enviados: {data}"
        self.result_label.config(text=self.shared_data)

    def _process_command(self):
        messagebox.showinfo("Información", f"Figura seleccionada: {self.shape_var.get()}")

    def _get_validated_command(self):
        """Validates and retrieves the text from the command entry."""
        data = self.shape_var.get().strip()
        if not data:
            messagebox.showerror("Error", "No se seleccionó una figura.")
            return None
        return data

    def _create_window(self, title, geometry):
        """Creates and returns a new Toplevel window."""
        new_window = tk.Toplevel(self.main_window)
        new_window.title(title)
        new_window.geometry(geometry)
        return new_window

    def _update_parameter_fields(self):
        pass


def open_graph_window(self):
    """Opens the graph window for displaying 3D shapes."""
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

        self._plot_3d_figure(selected_shape, parameters)


def open_results_window(self):
    """Opens the results window to display shared data."""
    if not _is_window_active(self.results_window):
        self.results_window = self._create_window(
            self.RESULTS_WINDOW_TITLE, self.RESULTS_WINDOW_GEOMETRY
        )
        self.result_label = ttk.Label(self.results_window, text=self.shared_data, wraplength=300)
        self.result_label.pack(pady=20)





def run(self):
    """Ejecuta el loop principal de la app"""
    self.main_window.mainloop()







def _plot_3d_figure(shape, params):
    """Generates and plots the selected 3D figure based on user parameters."""
    fig = plt.figure()
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
        face = float(params["lado"]) / 2
        r = [-face, face]
        for s, e in zip([r[0], r[1]], r):
            ax.plot3D([s, s], [r[0], r[1]], [e, e], color="g")

    elif shape == "Pirámide":
        base = float(params["base"]) / 2
        height = float(params["altura"])
        vertices = np.array([[0, 0, height], 
                             [-base, -base, 0], 
                             [base, -base, 0], 
                             [base, base, 0], 
                             [-base, base, 0]])
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="r", alpha=0.7)

    ax.set_title(f"Figura 3D: {shape}")
    plt.show()


def create_app():
    return MultiWindowApp()

if __name__ == "__main__":
    app = create_app()