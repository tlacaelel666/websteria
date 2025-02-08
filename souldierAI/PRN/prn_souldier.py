from datetime import datetime

class Norm:
    """
    Base class for a norm. Norms define an immutable behavior that will be applied.
    """

    def __init__(self, description: str):
        self.description = description  # Textual description of the norm

    def validate(self) -> bool:
        """
        Generic implementation for validating a norm.
        Returns True if the norm is met.
        """
        raise NotImplementedError("Base class should not be used directly.")


class Rule:
    """
    Base class for a rule. Generally refers to specific conditions (context).
       Rule o Regla es un conjunto de normas que
       forman una estructura sólida que por su naturaleza moldea las politics
    """
    def __init__(self, description: str):
        self.description = description  

    def validate(self) -> bool:
        """
        Generic validation implementation of rules.
        """
        raise NotImplementedError("Base class should not be used directly.")


class TimeRule(Rule):
    """
    Specific rule: Validate if we are between 02:00 and 11:00.
    """

    def __init__(self):
        super().__init__("The greeting 'Good morning' is only valid between 02:00 and 11:00.")

    def validate(self) -> bool:
        """
        Validates if the current time is within the 02:00 - 11:00 range.
        """
        current_hour = datetime.now().hour
        return 2 <= current_hour < 11


class MorningGreetingNorm(Norm):
    """
    Specific norm: Greeting "Good morning"
    """

    def __init__(self):
        super().__init__("Greet with 'Good morning'")
        self.greeted_today = False

    def validate(self) -> bool:
        """
        Validates if the greeting has already been performed today.
        """
        return not self.greeted_today

    def greet(self):
        print("Good morning!")
        self.greeted_today = True

    def reset_daily(self):
        self.greeted_today = False


class GreetingRule(Rule):
    """
    Intermediate rule that validates the greeting norm within the proper time.
    """
    def __init__(self):
        super().__init__("Validate the greeting norm and time.")
        self.time_rule = TimeRule()
        self.greeting_norm = MorningGreetingNorm()

    def validate(self) -> bool:
        """
        Validates that the greeting occurs within the allowed times.
        Returns True if both the time rule and the norm are valid.
        """
        return self.time_rule.validate() and self.greeting_norm.validate()

    def apply_greeting_norm(self):
        """
        Applies normalization to greeting-related text data. This function implements a specific
        normalization pipeline to standardize greetings and related conversational content
        in text. The normalization may involve text replacement, case conversion, trimming,
        or other text processing techniques that help streamline and unify the structure
        of a greeting phrase for downstream processing or analysis.

        Raises:
            ValueError: If the input text does not meet the expected format requirements
                or fails during the normalization process.
        """


class Policy:

    def __init__(self, description:str):
        self.description = description
    def enforce(self) -> None:
        """policy enforcement will usually be implemented in a subclass"""
        raise NotImplementedError("do not use base class directly.")


class Greetings(Policy):
    """specific policy for greetings"""
    def __init__(self):
        super().__init__("greetings: 'buenos dias, como va tu dia?' once at day until month long ")
        self.order_values = None
        self.greeting_norm = MorningGreetingNorm() # Use the MorningGreetingNorm
        self.greeted_today = False
        self.priority =  {'ethical': 2, 'legal': 1, 'status_quo': 0}
    def enforce(self) -> None:
        """implementation of policy Greetings, for the context and rules defined"""
        self.apply_policy()

    def apply_policy(self) -> None:
        """Apply the greeting policy with relevant checks and actions."""
        if self.can_greet():
            self.greeting_norm.greet() # Use the greet method from the norm
            self.greeted_today = True

    def can_greet(self) -> bool:
        """Check if greeting can proceed based on conditions."""
        return not self.greeted_today 

    def list_relevance(self):
        items = ['ethical', 'legal', 'status_quo']
        return sorted(items, key=lambda x: self.priority.get(x, 0), reverse=True)

    def define_priorities(self, order_values):
        self.order_values = {'ethical': 2, 'legal': 1, 'status_quo': 0}
        return sorted(order_values.keys(), key=lambda x: order_values[x], reverse=True)

    def reset_daily(self):
      """
      Resetea el estado diario para permitir un nuevo saludo al día siguiente.
      """
      self.greeting_norm.reset_daily() # Reset the norm's daily state
      self.greeted_today = False