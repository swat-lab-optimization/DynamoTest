import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

config = {
    "observation": {
        "type": "Kinematics",
        "absolute": True,
        "normalize": True,
        "vehicles_count": 2,
        "features_range": {
            "x": [200, 1400],
            "y": [4, 0],
            "vx": [19, 30],
            "vy": [-1.7, 1.7],
        },
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 2,
    "vehicles_count": 1,
    "controlled_vehicles": 1,
    "duration": 50,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # adversarial_behavior.AdversarialVehicle .IDMVehicle
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}
if __name__ == "__main__":
    # Create the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)

    model = DQN.load("model_checkpoints\\model_ego_dqn_test_1000_steps.zip", env=env)

    env = RecordVideo(env, video_folder="videos/test", episode_trigger=lambda e: True)

    env.unwrapped.set_record_video_wrapper(env)

    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering

    for videos in range(20):
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
