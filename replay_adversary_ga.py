import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, DQN

import highway_env  # noqa: F401
import numpy as np
from gymnasium.envs.registration import register
import json

import tyro
from adversrial_policies.deepq_policy import DeepQModel
from adversrial_policies.master_policy import MasterPolicy
from adversrial_policies.approach_policy import ApproachPolicy
from adversrial_policies.approach_policy_dqn import ApproachPolicyDQN
from adversrial_policies.change_lane import ChangeLanePolicy
from adversrial_policies.change_lane_dqn import ChangeLanePolicyDQN
from adversrial_policies.idm_policy import IdmPolicy
from datetime import datetime
from adversrial_policies.follow_policy import FollowPolicyPID
from rl_agents.agents.common.factory import agent_factory, load_agent_config, load_agent
import os
from common.utils import StatRecorder
from agents.dqn_agent_cleanrl import Args, DQNAgentCLRL
from envs.highway_env_adv import HighwayEnvAdversary
from agents.cmab_agent import CMabModel
from mabwiser.mab import MAB, LearningPolicy
from agents.random_agent import RandomAgent
import shutil
TRAIN = True
from ga.test_generator import GATester, GAInitTester

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


EPISODES = 1005
if __name__ == "__main__":
    runs = 10
    ego_types = [
        "baseline_defensive",
    ]
    algo = "ga"
    for ego_type in ego_types:
        all_tests = {}
        for run in range(runs):
            env_ego = gym.make(
                "highway-fast-v0", render_mode="rgb_array", config=config
            )
            env = gym.make("highwayadv-v0", render_mode="rgb_array", config=config_adv)
            env.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(2, 5), dtype=np.float32
            )



            available_actions = list(range(5))
            #model_adv = FollowPolicyPID(
            #    target_distance=10, front_follow=True
            #)  # IdmPolicy()

            if algo == "dqn":
                args = tyro.cli(Args)
                model_adv = DQNAgentCLRL(env_ego, args)
            elif algo == "cmab":
                model_adv = CMabModel(actions=available_actions, learning_policy=LearningPolicy.LinUCB(alpha=1.25, l2_lambda=1))
            elif algo == "random":
                model_adv = RandomAgent(env_ego)
            #model_adv = CMabModel(arms=available_actions, learning_policy=LearningPolicy.LinUCB(alpha=1.25, l2_lambda=1))
            elif algo == "ga":
                # dynamic_test_file = "stats\\RQ\\RQ1\\uc1\\rl_2025-10-18-4005-dqn_baseline_safe_dist\\extracted_tests.json"
                # fail_dict_file = "stats\\RQ\\RQ1\\uc1\\rl_2025-10-18-4005-dqn_baseline_safe_dist\\extracted_fail_configs.json"
                dynamic_test_file = "stats\\RQ\\RQ1\\uc2\\rl_2025-10-21-4005-dqn_baseline_defensive_uc2\\extracted_tests.json"
                fail_dict_file = "stats\\RQ\\RQ1\\uc2\\rl_2025-10-21-4005-dqn_baseline_defensive_uc2\\extracted_fail_configs.json"
                with open(dynamic_test_file, 'r') as f:
                     all_dynamic_tests = json.load(f)
                dynamic_tests = all_dynamic_tests[f"run_{run}"]
                with open(fail_dict_file, 'r') as f:
                    all_fail_dicts = json.load(f)
                fail_dicts = all_fail_dicts[f"run_{run}"]
                tester = GAInitTester(name="GATester", config="tester_config.yaml", dynamic_tests=dynamic_tests)
                tester.initialize()
            env_ego.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(5, 5), dtype=np.float32
            )

            #obs, info = env.reset()

            if ego_type == "baseline":
                model_json_file = "models\\dqn.json"
                adv_file = "out\\run_20250627-184624_22388\\checkpoint-best.tar"
                ego_file = "out\\run_20250627-174917_9328_ego\\checkpoint-best.tar"
                a_c_1 = load_agent_config(model_json_file)
                model_ego = agent_factory(env_ego, a_c_1)
                model_ego.load(ego_file)
            elif ego_type == "baseline_defensive":
                model_json_file = "models\\dqn.json"
                ego_file = "out\\run_20251020-125802_21988_ego_defensive\\checkpoint-best.tar"
                #ego_file = "out\\out\\run_20251026-135023_17256_ego_defensive_2\\checkpoint-best.tar"
                a_c_1 = load_agent_config(model_json_file)
                model_ego = agent_factory(env_ego, a_c_1)
            else:

                model_ego = DQN.load(
                    f"models/model_ego_dqn_01_05_25_{ego_type}.zip", env=env_ego
                )
        #     model_ego = DQN.load(
        #     f"models\\model_ego_dqn_26_05_25_aggressive.zip", env=env_ego
        # )
            #env.reset_failure_dict()
            env.load_failure_dict(fail_dicts)


            # model_adv.load("weights\\RL\\rl_2025-04-07-1000-dqn\\adversarial_vehicle_model_0")

            padding = np.zeros((3, 5))
            save_interval = 200
            add_info = f"{algo}_{ego_type}_ga_100_100_post_2"  # reward-2"
            experiment_name = f"{cur_date}-{EPISODES}-{add_info}"  #
            weights_folder = f"weights\\final_uc2\\rl_{experiment_name}\\run_{run}"
            stats_folder = f"stats\\final_uc2\\rl_{experiment_name}\\run_{run}"
            experiment_description = """Using the generated tests by RL and running the GA by varying the initial conditions. N off to 100, pop 100"""
            stat_recorder = StatRecorder(
                filepath=stats_folder,
                train=TRAIN,
                experiment_description=experiment_description,
            )
            env.trace_recorder.save_folder = stats_folder
            shutil.copyfile("tester_config.yaml", f"stats\\final_uc2\\rl_{experiment_name}\\tester_config.yaml")

            for ep in range(EPISODES):
                #env.trace_recorder_save_path = stats_folder
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
                        env.unwrapped.controlled_vehicles[i].position = np.array([test["init"][0], test['init'][1]])
                        #env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['ego_veh']['speed'])
                        env.unwrapped.controlled_vehicles[i].heading = test["init"][2]
                        #env.unwrapped.controlled_vehicles[i].speed = test["init"][3]
                        env.unwrapped.controlled_vehicles[i].target_lane_index = (test["init"][4])
                    elif i == 1:
                        env.unwrapped.controlled_vehicles[i].position = np.array([test['init'][5], test['init'][6]])
                        #env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['adv_veh']['speed'])
                        env.unwrapped.controlled_vehicles[i].heading = test['init'][7]
                       # env.unwrapped.controlled_vehicles[i].speed = test['init'][8]
                        env.unwrapped.controlled_vehicles[i].target_lane_index = (test['init'][9])
                i = 0
                adv_action_list = test['adv_actions']
                while not (done or truncated):
                    # Dispatch the observations to the model to get the tuple of actions

                    obs_ego = np.vstack((obs[0], padding))
                    obs_adv = obs[0]

                    action_ego = int(model_ego.plan(obs_ego)[0])

                    if algo == "ga":
                        action_adv = adv_action_list[i]
                    else:
                        action_adv = model_adv.predict(obs_adv, first=first)
                    action = ((action_ego),(action_adv))

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
                    if env.unwrapped.controlled_vehicles[0].crashed or i >= len(adv_action_list)-1:
                        done = True
                    obs = next_obs
                    first = False
                    #env.render()
                    total_reward += reward
                    i += 1
                print(f"Total reward: {total_reward}")
                if algo == "ga":
                    tester.tell_next(test, [total_reward*(-1)], "pass")

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
            save_path = os.path.join(stats_folder, f"search_stats.json")
            with open(save_path, "w") as f:
                json.dump(all_tests, f, indent=4)
