import json
import os
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
from dynasto.configs.agent_configs import config_adv, config_ego_test
from dynasto.ga.test_generator import GATester

TRAIN = True
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
    entry_point="dynasto.envs.highway_env_adv:HighwayEnvAdversary",
)

cur_date = datetime.now().strftime("%Y-%m-%d")


EPISODES = 4005
RENDER = False
if __name__ == "__main__":
    runs = 10
    algo = "ga"
    all_tests = {}
    ego_type = "use_case_1"
    for run in range(0, runs):
        env_ego = gym.make("highway-fast-v0", render_mode="rgb_array", config=config_ego_test)
        env_adv = gym.make("highwayadv-v0", render_mode="rgb_array", config=config_adv)
        env_adv.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2, 5), dtype=np.float32
        )

        available_actions = list(range(5))

        if algo == "dqn":
            args = tyro.cli(Args)
            model_adv = DQNAgentCLRL(env_ego, args)
        elif algo == "random":
            model_adv = RandomAgent(env_ego)
        elif algo == "ga":
            tester = GATester(name="GATester", config="adversarial_test_generation\\tester_config.yaml")
            tester.initialize()
        env_ego.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5, 5), dtype=np.float32
        )

        model_json_file = "src\\dynasto\\configs\\dqn.json"
        ego_file = f"ego_agents\\{ego_type}\\checkpoint-best.tar"
        a_c_1 = load_agent_config(model_json_file)
        model_ego = agent_factory(env_ego, a_c_1)
        model_ego.load(ego_file)

        env_adv.unwrapped.reset_failure_dict()

        padding = np.zeros((3, 5))
        save_interval = 200
        add_info = f"{algo}_{ego_type}"  # reward-2"
        experiment_name = f"{cur_date}-{EPISODES}-{add_info}"  #
        weights_folder = f"weights\\rl_{experiment_name}\\run_{run}"
        stats_folder = f"stats\\rl_{experiment_name}\\run_{run}"
        experiment_description = """ """
        stat_recorder = StatRecorder(
            filepath=stats_folder,
            train=TRAIN,
            experiment_description=experiment_description,
        )
        env_adv.unwrapped.trace_recorder.save_folder = stats_folder


        for ep in range(EPISODES):
            # env_advtrace_recorder_save_path = stats_folder
            obs, info = env_adv.reset()
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

                next_obs, reward, done, truncated, info = env_adv.step(action)
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
                if env_adv.unwrapped.controlled_vehicles[0].crashed:
                    done = True
                obs = next_obs
                first = False
                if RENDER:
                    env_adv.render()
                total_reward += reward
                i += 1
            print(f"Total reward: {total_reward}")
            if algo == "ga":
                tester.tell_next(test, [total_reward * (-1)], "pass")

            stat_recorder.save_stats(episode=ep, env=env_adv.unwrapped)
            # Save the model

            if algo != "ga":
                if TRAIN and ep % save_interval == 0:
                    model_path = os.path.join(weights_folder, f"adversary_{ep}")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    model_adv.save(model_path)

        env_adv.close()
        if algo == "ga":
            generated_tests, res = tester.get_results()
            all_tests[f"run{run}"] = generated_tests
            save_path = os.path.join(stats_folder, "search_stats.json")
            with open(save_path, "w") as f:
                json.dump(all_tests, f, indent=4)
