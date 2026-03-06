import gymnasium as gym
from agents.abstract_agent import AbstractAgent

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

class RandomAgent(AbstractAgent):
    def __init__(self, env):
        self.name = "RandomAgent"
        self.env = env  

    def predict(self, obs, first=False):
        actions = self.env.action_space.sample()
        return actions

    def update(self, obs, actions, next_obs, rewards, infos, terminations, truncations):
        pass

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass