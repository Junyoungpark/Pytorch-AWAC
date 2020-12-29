import numpy as np


class DiscreteRandomAgent:

    def __init__(self, num_actions: int):
        super(DiscreteRandomAgent, self).__init__()
        self.num_actions = num_actions

    def get_action(self, state=None):
        return np.random.randint(low=0, high=self.num_actions)
