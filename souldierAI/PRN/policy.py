class Policy:

    def __init__(self, norm.description:str):
        self.description = description
    def enforce(self) -> None:
        """policy enforcement will usually be implemented in a subclass"""
        raise NotImplementedError("do not use base class directly.")
class Greetings(Policy):
    """specific policy for greetings"""
    def __init__(self):
        super().__init__("greetings: 'buenos dias, como va tu dia?' once at day until month long ")
        self.order_values = None
        self.greeting_norm = 'buenos dias, como va tu dia?'
        self.greeted_today = False
        self.priority =  {'ethical': 2, 'legal': 1, 'status_quo': 0}
    def enforce(self) -> None:
        """implementation of policy Greetings, for the context and rules defined"""
        self.apply_policy()

    def apply_policy(self) -> None:
        """Apply the greeting policy with relevant checks and actions."""
        if self.can_greet():
            print("greeted today")
            self.greeted_today = True

    def can_greet(self) -> bool:
        """Check if greeting can proceed based on conditions."""
        return not self.greeted_today and isinstance(self.greeting_norm, str)

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
      self.greeted_today = False
