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
from dynasto.ga.test_generator import GAInitTester

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


EPISODES = 1005
if __name__ == "__main__":
    runs = 10
    algo = "ga"
    all_tests = {}
    ego_type = "use_case_1"
    for run in range(runs):
        env_ego = gym.make(
            "highway-fast-v0", render_mode="rgb_array", config=config_ego_test
        )
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
            if ego_type == "use_case_1":
                dynamic_test_file = "extracted_tests\\uc1\\extracted_tests.json"
                fail_dict_file = "extracted_tests\\uc1\\extracted_fail_configs.json"
            elif ego_type == "use_case_2":
                dynamic_test_file = "extracted_tests\\uc2\\extracted_tests.json"
                fail_dict_file = "extracted_tests\\uc2\\extracted_fail_configs.json"
            else:
                raise ValueError("Invalid ego type")
            with open(dynamic_test_file) as f:
                all_dynamic_tests = json.load(f)
            dynamic_tests = all_dynamic_tests[f"run_{run}"]
            with open(fail_dict_file) as f:
                all_fail_dicts = json.load(f)
            fail_dicts = all_fail_dicts[f"run_{run}"]
            tester = GAInitTester(
                name="GATester",
                config="adversarial_test_generation\\tester_config.yaml",
                dynamic_tests=dynamic_tests,
            )
            tester.initialize()
        env_ego.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5, 5), dtype=np.float32
        )

        # obs, info = env.reset()

        model_json_file = "src\\dynasto\\configs\\dqn.json"
        ego_file = f"ego_agents\\{ego_type}\\checkpoint-best.tar"
        a_c_1 = load_agent_config(model_json_file)
        model_ego = agent_factory(env_ego, a_c_1)
        model_ego.load(ego_file)
        env.unwrapped.load_failure_dict(fail_dicts)

        padding = np.zeros((3, 5))
        save_interval = 200
        add_info = f"{algo}_{ego_type}_dynasto"  # reward-2"
        experiment_name = f"{cur_date}-{EPISODES}-{add_info}"  #
        weights_folder = f"weights\\rl_{experiment_name}\\run_{run}"
        stats_folder = f"stats\\rl_{experiment_name}\\run_{run}"
        experiment_description = """."""
        stat_recorder = StatRecorder(
            filepath=stats_folder,
            train=TRAIN,
            experiment_description=experiment_description,
        )
        env.unwrapped.trace_recorder.save_folder = stats_folder
        for ep in range(EPISODES):

            obs, info = env.reset()
            obs_ego, info_ego = env_ego.reset()
            print(f"Episode {ep}")
            total_reward = 0

            done = truncated = False
            first = True
            if algo == "ga":
                test = tester.ask_next()
                valid, fitness, test = tester.verify_next(test)

            for i in range(len(env.unwrapped.controlled_vehicles)):
                if i == 0:
                    env.unwrapped.controlled_vehicles[i].position = np.array(
                        [test["init"][0], test["init"][1]]
                    )
                    # env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['ego_veh']['speed'])
                    env.unwrapped.controlled_vehicles[i].heading = test["init"][2]
                    env.unwrapped.controlled_vehicles[i].speed = test["init"][3]
                    env.unwrapped.controlled_vehicles[i].target_lane_index = test[
                        "init"
                    ][4]
                elif i == 1:
                    env.unwrapped.controlled_vehicles[i].position = np.array(
                        [test["init"][5], test["init"][6]]
                    )
                    # env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['adv_veh']['speed'])
                    env.unwrapped.controlled_vehicles[i].heading = test["init"][7]
                    env.unwrapped.controlled_vehicles[i].speed = test['init'][8]
                    env.unwrapped.controlled_vehicles[i].target_lane_index = test[
                        "init"
                    ][9]
            i = 0
            adv_action_list = test["adv_actions"]
            while not (done or truncated):
                # Dispatch the observations to the model to get the tuple of actions

                obs_ego = np.vstack((obs[0], padding))
                obs_adv = obs[0]

                action_ego = int(model_ego.plan(obs_ego)[0])

                if algo == "ga":
                    action_adv = adv_action_list[i]
                else:
                    action_adv = model_adv.predict(obs_adv, first=first)
                action = ((action_ego), (action_adv))

                next_obs, reward, done, truncated, info = env.step(action)
                next_obs_adv = next_obs[0]

                if algo != "ga":
                    model_adv.update(
                        obs_adv,
                        action[1],
                        next_obs_adv,
                        float(reward),
                        info,
                        done,
                        truncated,
                    )
                if (
                    env.unwrapped.controlled_vehicles[0].crashed
                    or i >= len(adv_action_list) - 1
                ):
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
        generated_tests, res = tester.get_results()
        all_tests[f"run{run}"] = generated_tests
        save_path = os.path.join(stats_folder, "search_stats.json")
        with open(save_path, "w") as f:
            json.dump(all_tests, f, indent=4)
