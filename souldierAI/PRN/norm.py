from PRN.prn_souldier import GreetingRule
from logic.bayes_logic import MorningGreetingNorm
from logic.validator import Validator


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


from datetime import datetime


class Rule:
    """
    Base class for a rule. Generally refers to specific conditions (context).
       Rule o Regla es un conjunto de normas que
       forman una estructura solida que por su baturaleza moldea las politicas
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
        Uses a `Validator` to perform validation logic.
        """
        validator = Validator()
        return validator.validate_time(2, 11)

    class GreetingRule(Rule):
        """
        Intermediate rule that validates the greeting norm within the proper time.
        """

        def __init__(self, description: str = "Validate the greeting norm and time."):
            super().__init__(description)  # Initialize the base class with a description
            self.time_rule = TimeRule()  # Assign a new TimeRule instance
            self.greeting_norm = MorningGreetingNorm()  # Initialize the greeting norm

    def apply_greeting_norm(self):
        """
        Applies the greeting norm policy by using the greeting norm enforcement logic.
        """
        print("Buenos días, ¿cómo va tu día?")

    def validate(self) -> bool:
        """
        Validates that the greeting occurs within the allowed times.
        Returns True if both the time rule and the norm are valid.
        """
        return self.time_rule.validate() and self.greeting_norm.validate()


if __name__ == "__main__":
    rule = GreetingRule("Validate the greeting norm and time.")
    if rule.validate():
        rule.apply_greeting_norm()
    else:
        print("The greeting does not comply with the norm or allowed time.")
