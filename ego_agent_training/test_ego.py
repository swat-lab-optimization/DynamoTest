import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

from dynasto.configs.agent_configs import ego_config

if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config=ego_config)

    model = DQN.load("model_checkpoints\\ego_dqn_defensive_500_steps.zip", env=env)

    env = RecordVideo(env, video_folder="videos/test", episode_trigger=lambda e: True)

    env.unwrapped.set_record_video_wrapper(env)

    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering

    for _ in range(20):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
