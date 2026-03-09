import json

from agents.abstract_agent import AbstractAgent


class FileAgent(AbstractAgent):
    """Dummy code for an RL algorithm, which predicts an action from an observation,
    and update its model from observed transitions."""

    def __init__(self, scenario_file: str = None):

        super().__init__("FileAgent")
        assert scenario_file is not None, "Scenario file must be provided"
        with open(scenario_file) as f:
            self.scenario = json.load(f)

    def predict(self, obs, step, veh_type: str):
        """Predicts the action based on the scenario file and step number."""

        if veh_type == "ego":
            try:
                action = self.scenario[str(step)]["ego_action"]
            except KeyError:
                # If the step is not found, return a default action (e.g., 1- idle)
                action = 1
        elif veh_type == "adv":
            try:
                action = self.scenario[str(step)]["adv_action"]
            except KeyError:
                action = 1
        else:
            raise ValueError("Vehicle type must be 'ego' or 'adv'")
        return action

    def update(self, obs, action, next_obs, reward, info, done, truncated):
        """Update method is not used in this agent, but defined for compatibility."""
        pass
