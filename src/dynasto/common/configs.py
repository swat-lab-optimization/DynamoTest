config_adv = {
    "controlled_vehicles": 2,
    "lanes_count": 2,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
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
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
}

config = {
    "controlled_vehicles": 1,
    "lanes_count": 2,
    "vehicles_count": 1,
    "observation": {"type": "Kinematics"},
    "action": {
        "type": "DiscreteMetaAction",
    },
}
