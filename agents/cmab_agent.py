import numpy as np
from agents.abstract_agent import AbstractAgent
from typing import List
from mabwiser.mab import MAB, LearningPolicy
import pickle

class CMabModel(AbstractAgent):
    """Dummy code for an RL algorithm, which predicts an action from an observation,
    and update its model from observed transitions."""

    def __init__(self, actions: List[int], learning_policy: LearningPolicy = None):
        self.mab = MAB(arms=actions, learning_policy=learning_policy)

        assert len(actions) > 0, "Actions list must not be empty"
        assert learning_policy is not None, "Learning policy must be defined"

        super().__init__("CMabModel")

    def predict(self, obs, first=False):
        context = [float(x) for x in obs[0]] + [float(x) for x in obs[1]]
        if first:
            return np.random.choice(self.mab.arms)
        else:
            return self.mab.predict([context])

    def update(self, obs, action, next_obs, reward, info, done, truncated):
        context = [float(x) for x in next_obs[0]] + [float(x) for x in obs[1]]
        try:
            self.mab.partial_fit(
                decisions=[action], rewards=[reward], contexts=[context]
            )
        except Exception as e:
            print("Error")
            print(e)

    def save(self, filename="adversarial_vehicle_mab.pkl"):

        with open(filename, 'wb') as f:
            pickle.dump(self.mab, f)
        print(f"Adversarial vehicle MAB saved to {filename}")

    def load(self, filename="adversarial_vehicle_mab.pkl"):
        with open(filename, "rb") as f:
            loaded_mab = pickle.load(f)
        self.mab = loaded_mab