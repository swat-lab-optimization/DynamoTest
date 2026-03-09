import json
import os
import shutil
from datetime import datetime

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import tyro
from gymnasium.envs.registration import register
from rl_agents.agents.common.factory import agent_factory, load_agent_config

from dynasto.agents.dqn_agent_cleanrl import Args, DQNAgentCLRL
from dynasto.agents.random_agent import RandomAgent
from dynasto.common.utils import StatRecorder
from dynasto.ga.test_generator import GATester

TRAIN = True
config_adv = {
    "controlled_vehicles": 2,
    "lanes_count": 2,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "absolute": True,
            "normalize": True,
            "vehicles_count": 2,  # Error if using 2: Unexpected observation shape (2, 5) for Box environment,
            "features_range": {
                "x": [100, 1400],
                "y": [0, 4],
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
        "type": "DiscreteMetaAction",
    },
}
"""
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
"""
# Dummy RL algorithm
register(
    id="highwayadv-v0",
    entry_point="envs.highway_env_adv:HighwayEnvAdversary",
)

cur_date = datetime.now().strftime("%Y-%m-%d")


EPISODES = 4005
if __name__ == "__main__":
    runs = 10
    ego_types = [
        "baseline_defensive",
    ]
    algo = "dqn"
    all_tests = {}
    ego_type = "use_case_1"
    for run in range(0, runs):
        env_ego = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)
        env = gym.make("highwayadv-v0", render_mode="rgb_array", config=config_adv)
        env.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2, 5), dtype=np.float32
        )

        available_actions = list(range(5))

        if algo == "dqn":
            args = tyro.cli(Args)
            model_adv = DQNAgentCLRL(env_ego, args)
        elif algo == "random":
            model_adv = RandomAgent(env_ego)
        elif algo == "ga":
            tester = GATester(name="GATester", config="tester_config.yaml")
            tester.initialize()
        env_ego.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5, 5), dtype=np.float32
        )

        model_json_file = "models\\dqn.json"
        # adv_file = "out\\run_20250627-184624_22388\\checkpoint-best.tar"
        ego_file = f"ego_agents\\{ego_type}\\checkpoint-best.tar"
        a_c_1 = load_agent_config(model_json_file)
        model_ego = agent_factory(env_ego, a_c_1)
        model_ego.load(ego_file)

        env.reset_failure_dict()

        padding = np.zeros((3, 5))
        save_interval = 200
        add_info = f"{algo}_{ego_type}_new_uc2_no_novelty"  # reward-2"
        experiment_name = f"{cur_date}-{EPISODES}-{add_info}"  #
        weights_folder = f"weights\\final_26_oct_uc2\\rl_{experiment_name}\\run_{run}"
        stats_folder = f"stats\\final_26_oct_uc2\\rl_{experiment_name}\\run_{run}"
        experiment_description = """New defensive agent, nonovelty added. """
        stat_recorder = StatRecorder(
            filepath=stats_folder,
            train=TRAIN,
            experiment_description=experiment_description,
        )
        env.trace_recorder.save_folder = stats_folder
        shutil.copyfile(
            "envs\\highway_env_adv.py",
            f"stats\\final_26_oct_uc2\\rl_{experiment_name}\\highway_env_adv.py",
        )
        if algo == "dqn":
            shutil.copyfile(
                "agents\\dqn_agent_cleanrl.py",
                f"stats\\final_26_oct_uc2\\rl_{experiment_name}\\dqn_agent_cleanrl.py",
            )

        for ep in range(EPISODES):
            # env.trace_recorder_save_path = stats_folder
            obs, info = env.reset()
            obs_ego, info_ego = env_ego.reset()
            print(f"Episode {ep}")
            total_reward = 0

            done = truncated = False
            first = True
            if algo == "ga":
                test = tester.ask_next()
                valid, fitness, test = tester.verify_next(test)
            i = 0
            while not (done or truncated):
                # Dispatch the observations to the model to get the tuple of actions

                obs_ego = np.vstack((obs[0], padding))
                obs_adv = obs[0]

                action_ego = int(model_ego.plan(obs_ego)[0])

                # action_ego = int(model_ego.predict(obs_ego)[0])

                if algo == "ga":
                    action_adv = test[i]
                else:
                    action_adv = model_adv.predict(obs_adv)
                action = ((action_ego), (action_adv))

                next_obs, reward, done, truncated, info = env.step(action)
                next_obs_adv = next_obs[0]

                if algo != "ga":
                    if algo == "sac":
                        model_adv.add_transition(
                            obs_adv,
                            next_obs_adv,
                            action[1],
                            float(reward),
                            done,
                            info,
                        )
                        model_adv.update()
                    else:
                        model_adv.update(
                            obs_adv,
                            action[1],
                            next_obs_adv,
                            float(reward),
                            info,
                            done,
                            truncated,
                        )
                if env.unwrapped.controlled_vehicles[0].crashed:
                    done = True
                obs = next_obs
                first = False
                # env.render()
                total_reward += reward
                i += 1
            print(f"Total reward: {total_reward}")
            if algo == "ga":
                tester.tell_next(test, [total_reward * (-1)], "pass")

            stat_recorder.save_stats(episode=ep, env=env.unwrapped)
            # Save the model

            if algo != "ga":
                if TRAIN and ep % save_interval == 0:
                    model_path = os.path.join(weights_folder, f"adversary_{ep}")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    model_adv.save(model_path)

        env.close()
        if algo == "ga":
            generated_tests, res = tester.get_results()
            all_tests[f"run{run}"] = generated_tests
            save_path = os.path.join(stats_folder, "search_stats.json")
            with open(save_path, "w") as f:
                json.dump(all_tests, f, indent=4)
