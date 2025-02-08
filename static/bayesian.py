class BayesLogic:
    EPSILON = 1e-6
    HIGH_ENTROPY_THRESHOLD = 0.8
    HIGH_COHERENCE_THRESHOLD = 0.6
    ACTION_THRESHOLD = 0.5

def __init__(self):
    self.EPSILON = 1e-6
    self.HIGH_ENTROPY_THRESHOLD = 0.8
    self.HIGH_COHERENCE_THRESHOLD = 0.6
    self.ACTION_THRESHOLD = 0.5

def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
    """Calculate the posterior probability using Bayes' theorem."""
    prior_b = prior_b if prior_b != 0 else self.EPSILON
    return (conditional_b_given_a * prior_a) / prior_b

def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
    """Calculate a conditional probability given joint probability and prior."""
    prior = prior if prior != 0 else self.EPSILON
    return joint_probability / prior

def calculate_high_entropy_prior(self, entropy: float) -> float:
    """Get prior based on entropy value."""
    return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

def calculate_high_coherence_prior(self, coherence: float) -> float:
    """Get prior based on coherence value."""
    return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
    """Calculate the joint probability of A and B based on coherence and action."""
    if coherence > self.HIGH_COHERENCE_THRESHOLD:
        if action == 1:
            return prn_influence * 0.8 + (1 - prn_influence) * 0.2
        else:
            return prn_influence * 0.1 + (1 - prn_influence) * 0.7
    return 0.3

def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, prn_influence: float, action: int) -> dict:
    """Calculate probabilities and select an action based on entropy, coherence, PRN, and action."""
    high_entropy_prior = self.calculate_high_entropy_prior(entropy)
    high_coherence_prior = self.calculate_high_coherence_prior(coherence)

    conditional_b_given_a = prn_influence * 0.7 + (1 - prn_influence) * 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2
    posterior_a_given_b = self.calculate_posterior_probability(high_entropy_prior, high_coherence_prior, conditional_b_given_a)

    joint_probability_ab = self.calculate_joint_probability(coherence, action, prn_influence)
    conditional_action_given_b = self.calculate_conditional_probability(joint_probability_ab, high_coherence_prior)

    action_to_take = 1 if conditional_action_given_b > self.ACTION_THRESHOLD else 0

    return {
        "action_to_take": action_to_take,
        "high_entropy_prior": high_entropy_prior,
        "high_coherence_prior": high_coherence_prior,
        "posterior_a_given_b": posterior_a_given_b,
        "conditional_action_given_b": conditional_action_given_b,
    }