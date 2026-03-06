import gymnasium as gym
from stable_baselines3 import DQN
import os
import highway_env  # noqa: F401
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# highway_env._register_highway_envs()
TRAIN = True
config = {
    "observation": {
        "type": "Kinematics",
        "absolute": True,
        "normalize": True,
        "vehicles_count": 2,
        "features_range": {
            "x": [100, 1400],
            "y": [0, 4],
            "vx": [19, 30],
            "vy": [-1.7, 1.7],
        },
    },
    "action": {
        "type": "DiscreteMetaAction",  # "DiscreteMetaAction",
    },
    "lanes_count": 2,
    "vehicles_count": 1,
    "duration": 40,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.DefensiveVehicle",  # IDMVehicle",DefensiveVehicle #AggressiveVehicle", #
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}


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

    model_name = "model_ego_dqn_26_05_25_defensive"

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

    """
    model = PPO(
        "MlpPolicy",
        env,

        ent_coef=0.01,
        tensorboard_log=tensorboard_save_path,
        device="cuda",  # Use GPU if available
    )"""

    checkpoint_callback = CheckpointCallback(
        save_freq=500,  # Save every 10000 steps (adjust as needed)
        save_path="./model_checkpoints/",  # Directory to save checkpoints
        name_prefix=model_name,  # Prefix for the checkpoint files
    )

    if TRAIN:
        model.learn(
            total_timesteps=EPISODES * config["duration"]
        )  # , callback = RenderCallback(render_freq=1) , callback=checkpoint_callback
        model.save(model_save_path)
        # del model
