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
