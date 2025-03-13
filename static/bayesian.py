import numpy as np
import mpmath
import scipy.stats as stats
from typing import List, Dict, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Enumeración para tipos de acciones
class ActionType(Enum):
    CONSERVATIVE = auto()
    MODERATE = auto()
    AGGRESSIVE = auto()

@dataclass
class ProbabilisticModel:
    """
    Modelo probabilístico base con múltiple precisión
    """
    prior_probability: float = 0.5
    precision: int = 100  # Precisión de cálculo
    
    def set_precision(self, precision: int):
        """Configura la precisión de cálculo"""
        mpmath.mp.dps = precision

    def calculate_entropy(self, probabilities: List[float]) -> float:
        """
        Calcula la entropía de Shannon
        
        Args:
            probabilities (List[float]): Lista de probabilidades
        
        Returns:
            float: Valor de entropía
        """
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

    def bayesian_update(self, 
                         prior: float, 
                         likelihood: float, 
                         evidence: float) -> float:
        """
        Actualización bayesiana con manejo de precisión
        
        Args:
            prior (float): Probabilidad previa
            likelihood (float): Verosimilitud
            evidence (float): Evidencia
        
        Returns:
            float: Probabilidad posterior
        """
        try:
            posterior = (likelihood * prior) / (evidence + 1e-10)
            return max(0, min(posterior, 1))
        except Exception as e:
            logging.error(f"Error en actualización bayesiana: {e}")
            return prior

@dataclass
class BayesLogic:
    """
    Lógica bayesiana avanzada con múltiple precisión
    """
    # Constantes de umbral
    EPSILON: float = 1e-6
    HIGH_ENTROPY_THRESHOLD: float = 0.8
    HIGH_COHERENCE_THRESHOLD: float = 0.6
    ACTION_THRESHOLD: float = 0.5

    # Configuraciones internas
    probabilistic_model: ProbabilisticModel = field(
        default_factory=ProbabilisticModel
    )
    
    # Registro de estados y acciones
    action_history: List[Dict] = field(default_factory=list)
    
    def calculate_conditional_probability(
        self, 
        event_a: float, 
        event_b: float
    ) -> float:
        """
        Calcula probabilidad condicional
        
        Args:
            event_a (float): Probabilidad de evento A
            event_b (float): Probabilidad de evento B
        
        Returns:
            float: Probabilidad condicional
        """
        return (event_a * event_b) / (event_b + self.EPSILON)

    def select_action(
        self, 
        probabilities: List[float], 
        custom_threshold: float = None
    ) -> ActionType:
        """
        Selecciona acción basada en probabilidades
        
        Args:
            probabilities (List[float]): Probabilidades de eventos
            custom_threshold (float, optional): Umbral personalizado
        
        Returns:
            ActionType: Tipo de acción seleccionada
        """
        threshold = custom_threshold or self.ACTION_THRESHOLD
        entropy = self.probabilistic_model.calculate_entropy(probabilities)
        
        if entropy > self.HIGH_ENTROPY_THRESHOLD:
            return ActionType.CONSERVATIVE
        
        max_prob = max(probabilities)
        
        if max_prob > threshold:
            return ActionType.AGGRESSIVE
        else:
            return ActionType.MODERATE

    def advanced_bayesian_inference(
        self, 
        prior_data: List[float], 
        likelihood_data: List[float]
    ) -> Dict[str, Union[float, ActionType]]:
        """
        Inferencia bayesiana avanzada
        
        Args:
            prior_data (List[float]): Datos previos
            likelihood_data (List[float]): Datos de verosimilitud
        
        Returns:
            Dict: Resultados de la inferencia
        """
        # Cálculo de probabilidades
        prior = np.mean(prior_data)
        likelihood = np.mean(likelihood_data)
        evidence = np.std(likelihood_data)
        
        # Actualización bayesiana
        posterior = self.probabilistic_model.bayesian_update(
            prior, likelihood, evidence
        )
        
        # Selección de acción
        action = self.select_action([prior, likelihood, posterior])
        
        # Registro de la acción
        result = {
            "prior": prior,
            "likelihood": likelihood,
            "posterior": posterior,
            "action": action,
            "entropy": self.probabilistic_model.calculate_entropy([prior, likelihood, posterior])
        }
        
        self.action_history.append(result)
        return result

    def analyze_action_history(self) -> Dict:
        """
        Analiza el historial de acciones
        
        Returns:
            Dict: Estadísticas del historial de acciones
        """
        if not self.action_history:
            return {}
        
        actions = [entry['action'] for entry in self.action_history]
        posteriors = [entry['posterior'] for entry in self.action_history]
        
        return {
            "total_actions": len(actions),
            "action_distribution": {
                action: actions.count(action) / len(actions) 
                for action in ActionType
            },
            "average_posterior": np.mean(posteriors),
            "posterior_variance": np.var(posteriors)
        }

# Ejemplo de uso
def main():
    # Configuración del modelo
    bayes_logic = BayesLogic()
    
    # Datos de ejemplo
    prior_data = [0.3, 0.4, 0.5]
    likelihood_data = [0.6, 0.7, 0.8]
    
    # Realizar inferencia bayesiana
    result = bayes_logic.advanced_bayesian_inference(
        prior_data, likelihood_data
    )
    
    print("Resultado de Inferencia:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    # Analizar historial de acciones
    history_analysis = bayes_logic.analyze_action_history()
    print("\nAnálisis de Historial:")
    for key, value in history_analysis.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()