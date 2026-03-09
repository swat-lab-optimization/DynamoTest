import os

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from dynasto.configs.agent_configs import config

# highway_env._register_highway_envs()
TRAIN = True


class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True


EPISODES = 4000
if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)
    # env.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 5), dtype=np.float32)
    obs, info = env.reset()

    model_name = "model_ego_dqn_test"

    tensorboard_save_path = os.path.join("tensorboard_logs", model_name)
    # Train the model
    model_save_path = os.path.join("models", model_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=150000,
        learning_starts=200,
        batch_size=64,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        verbose=1,
        tensorboard_log=tensorboard_save_path,
        device="cuda",  # Use GPU if available
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500,  # Save every 10000 steps (adjust as needed)
        save_path="./model_checkpoints/",  # Directory to save checkpoints
        name_prefix=model_name,  # Prefix for the checkpoint files
    )

    if TRAIN:
        model.learn(
            total_timesteps=EPISODES * config["duration"],
            callback=[RenderCallback(render_freq=1), checkpoint_callback],
        )  #
        model.save(model_save_path)
        # del model
