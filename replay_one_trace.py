import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, DQN
from gymnasium.envs.registration import register
import highway_env  # noqa: F401
import numpy as np
import json
from adversrial_policies.deepq_policy import DeepQModel
from adversrial_policies.master_policy import MasterPolicy
from adversrial_policies.approach_policy import ApproachPolicy
from adversrial_policies.approach_policy_dqn import ApproachPolicyDQN
from adversrial_policies.approach_policy_pid import ApproachPolicyPID
from adversrial_policies.follow_policy import FollowPolicyPID
from adversrial_policies.change_lane import ChangeLanePolicy
from adversrial_policies.change_lane_dqn import ChangeLanePolicyDQN
from datetime import datetime
from adversrial_policies.master_policy_HRL import MasterPolicyHRL, MasterPolicyHRLRandom
import os
from common.utils import StatRecorder
from common.tracers import CutInTracer
from common.behavior_evaluator import BehaviorEvaluator
from common.run_evaluator import RunEvaluator

from agents.file_agent import FileAgent
from adversrial_policies.change_lane_left import ChangeLaneLeftPolicy
from adversrial_policies.change_lane_right import ChangeLaneRightPolicy
from adversrial_policies.accelerate import AcceleratePolicy
from adversrial_policies.brake import BrakePolicy
from datetime import datetime
import time
from common.tracer_monitor import TracerMonitor
from common.trace_analyzer import TraceAnalyzer
import sys
from common.trace_recorder import TraceRecorder
from envs.highway_env_adv import HighwayEnvAdversary
from common.tracers import (
    FrontSameLaneTracer, CutOutTracer, # 1
    FrontSameLaneTracer, FrontSlowDownSameLaneTracer,

    BehindSameLaneTracer, BehindSpeedUpTracer, # 2

    FrontDifferentLaneTracer, CutInTracer, # 3 ego lane 0
    FrontDifferentLaneTracer, FrontSlowDownDifferentLaneTracer, # ego lane 0

    SideTracer, CutInTracer, # 4 ego lane 0

    BehindDifferentLaneTracer, CutInTracer, # 5 ego lane 0
    BehindDifferentLaneTracer, BehindSpeedUpTracer, #ego lane 0

    FrontDifferentLaneTracer, FrontSlowDownDifferentLaneTracer, # 6 ego lane 1
    FrontDifferentLaneTracer, CutInTracer, # ego lane 1

    SideTracer, CutInTracer, # 7 ego lane 1

    BehindDifferentLaneTracer, CutInTracer, # 8 ego lane 1
    BehindDifferentLaneTracer, BehindSpeedUpTracer, #ego lane 1

    CutInSideTracer, 
    EgoCutInTracer,
    EgoCutInSideTracer,
    EgoCutOutTracer
)

TRAIN = False

# highway_env._register_highway_envs()
config_adv = {
    "controlled_vehicles": 2,
    "lanes_count": 2,
    "duration": 40,
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
        "vehicles_count": 2,
        "absolute": True,
        "normalize": True,
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
}

ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

# Dummy RL algorithm

register(
    id="highwayadv-v0",
    entry_point=HighwayEnvAdversary,#"envs.highway_env_adv:HighwayEnvAdversary",
)

EPISODES = 30
if __name__ == "__main__":
    env_ego = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)
    env = gym.make("highwayadv-v0", render_mode="rgb_array", config=config_adv)
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 5), dtype=np.float32)
   # env._init()
    # env.configure(config)

    obs, info = env.reset()
    obs_ego, info_ego = env_ego.reset()

    # Create the model
    # model_ego =  DQN.load("models\\model_dqn_fin_5_dec_24", env=env_ego)
    ego_type = "idm"  # stats\RL\rl_2025-05-01-2005-single_agent_idm_approach\run_0
    # model_ego =  DQN.load(f"models\\model_ego_dqn_01_05_25_{ego_type}.zip", env=env_ego)
    #model_ego = DQN.load(f"models\\model_ego_dqn_26_05_25_aggressive.zip", env=env_ego)
    # model_ego  = DQN.load("models\\model.zip", env=env_ego)


    available_actions = list(range(5))

    # mab = MAB(arms=available_actions, learning_policy=LearningPolicy.LinUCB(alpha=2, l2_lambda=1))
    # model_adv = CMabModel(mab)
    #model_adv = ChangeLanePolicy()

    #weights_file = "weights\\RL\\rl_2025-06-26-2005-single_agent_cut_in_aggressive_req1-v2\\run_1\\adversary_2000"

    #video_folder = weights_file.replace("weights", "stats")
    folder =  int(sys.argv[1]) # 12 #
    recording_episode =  int(sys.argv[2]) # 12 #

    video_folder = f"stats\\final_22_oct_uc2_tmp\\rl_2025-10-25-4005-dqn_baseline_defensive_uc2_ga_200_&rl_2.2_no_nov\\run_10\\{folder}"
    #recording_init_file = f"{video_folder}\\scenario_init_episode_{recording_episode}.json"
    recording_file = f"{video_folder}\\scenario_trace_episode_recording_{recording_episode}.json"
    #tracer_monitor_file = f"{video_folder}\\tracer_monitor_pred_{recording_episode}.json"
    # with open(recording_init_file, 'r') as f:
    #     init_config = json.load(f)
    with open(recording_file, 'r') as f:
        recording = json.load(f)

    init_config = recording["0"]

    
    model_adv = FileAgent(scenario_file=recording_file)
    env = RecordVideo(env, video_folder=video_folder,
              episode_trigger=lambda e: True, name_prefix="video-replay")  # record all episodes

# Provide the video recorder to the wrapped environment
# so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)
    env.episode_id = recording_episode


    padding = np.zeros((3, 5))
    save_interval = 200
    cur_date = datetime.now().strftime("%Y-%m-%d")
    add_info = f"single_agent_{ego_type}_replay_baseline"  # reward-2"
    experiment_name = f"{cur_date}-{EPISODES}-{add_info}"  #
    stats_folder = f"stats\\RL\\rl_{experiment_name}_eval"
    experiment_description = """In this experiment we evaluate the signle agent"""
    stat_recorder = StatRecorder(
        filepath=stats_folder,
        train=TRAIN,
        experiment_description=experiment_description,
    )
    tracer = CutInTracer()
    #beh_eval = BehaviorEvaluator()
    #run_evaluator = RunEvaluator()
    tracer_monitor = TracerMonitor([EgoCutInSideTracer(),EgoCutInTracer(), EgoCutOutTracer(), CutInSideTracer(), CutOutTracer(), CutInTracer(), FrontSlowDownSameLaneTracer(), FrontSlowDownDifferentLaneTracer()])
    trace_analyzer = TraceAnalyzer()
    ep = recording_episode
    obs, info = env.reset()
    print(f"Episode {ep}")

    trace_recorder = TraceRecorder(
    save_folder=video_folder, episode= recording_episode
    )
    #video_annotator.save_trace_init(env.unwrapped.controlled_vehicles[0], env.unwrapped.controlled_vehicles[1])
    for i in range(len(env.unwrapped.controlled_vehicles)):
        if i == 0:
            env.unwrapped.controlled_vehicles[i].position = np.array([init_config['ego_x'], init_config['ego_lane']])
            #env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['ego_veh']['speed'])
            env.unwrapped.controlled_vehicles[i].heading = init_config['ego_heading']
            env.unwrapped.controlled_vehicles[i].target_lane_index = tuple(init_config['ego_target_lane'])
        elif i == 1:
            env.unwrapped.controlled_vehicles[i].position = np.array([init_config['adv_x'], init_config['adv_lane']])
            #env.unwrapped.controlled_vehicles[i].speed = np.array(init_config['adv_veh']['speed'])
            env.unwrapped.controlled_vehicles[i].heading = init_config['adv_heading']
            env.unwrapped.controlled_vehicles[i].target_lane_index = tuple(init_config['adv_target_lane'])

    total_reward = 0

    done = truncated = False
    first = True
    step = 0
    # tracer.update(
    #     env.unwrapped.controlled_vehicles[0],
    #     env.unwrapped.controlled_vehicles[1],
    # )
    while not (done or truncated):

        obs_ego = obs[0]  # np.vstack((obs[0], padding))  #
        obs_adv = np.append(obs[0][0][1:], obs[0][1][1:])

        action_ego = model_adv.predict(obs_ego, step, "ego")

        action_adv = model_adv.predict(
            obs_adv, step, "adv",
        )
        action = (int(action_ego), int(action_adv))
        #print(f"Action ego: {[action[0]]}, Action adv: {[action[1]]}, step: {step}")

        next_obs, reward, done, truncated, info = env.step(action)
        # tracer.update(
        #     env.unwrapped.controlled_vehicles[0],
        #     env.unwrapped.controlled_vehicles[1],
        # )
        print(f"Adv speed {env.unwrapped.controlled_vehicles[1].velocity}, Ego speed {env.unwrapped.controlled_vehicles[0].velocity}")
        print(f"Ego position {env.unwrapped.controlled_vehicles[0].position}, Adv position {env.unwrapped.controlled_vehicles[1].position}")
        print(f"Adv lane index {env.unwrapped.controlled_vehicles[1].lane_index[2]}, Ego lane index {env.unwrapped.controlled_vehicles[0].lane_index[2]}")
        print(f"Adv heading {env.unwrapped.controlled_vehicles[1].heading}, Ego heading {env.unwrapped.controlled_vehicles[0].heading}")
        print(f"Adv - ego distance {env.unwrapped.controlled_vehicles[1].position[0] - env.unwrapped.controlled_vehicles[0].position[0]}, step: {step}")
        print(f"--------------------------------------------------------")
        #tracer_monitor.monitor_step(tracer.input_trace)
        #trace_recorder.update_trace(ego_veh=env.unwrapped.controlled_vehicles[0], ego_action=action[0], adv_veh=env.unwrapped.controlled_vehicles[1], adv_action=action[1])
        #video_annotator.update_trace(ego_veh=env.unwrapped.controlled_vehicles[0], ego_action=action[0], adv_veh=env.unwrapped.controlled_vehicles[1], adv_action=action[1])
        next_obs_adv = np.append(next_obs[0][0][1:], next_obs[0][1][1:])
        if env.unwrapped.controlled_vehicles[0].crashed or step >= len(model_adv.scenario) -1:
            # print("Ego crashed")
            done = True
        obs = next_obs
        first = False

        env.render()
        # time.sleep(0.5)
        total_reward += reward
        step += 1

    #print(f"Tracer monitor: {tracer_monitor.tracer_dict}")
    print(f"Total reward: {total_reward}")
    print(f"Total steps: {step}")
    #trace_results = tracer.evaluate(tracer.input_trace)
    #tracer_monitor.monitor_episode(tracer.input_trace)
    
    #tracer_monitor.save(tracer_monitor_file)
    #trace_analyzer.analyze(tracer_monitor.tracer_dict, trace_recorder.all_frames_dict)
    #trace_recorder.save_trace()
    
    #beh_eval.record_behavior()
    #tracer.reset()

    env.close()
