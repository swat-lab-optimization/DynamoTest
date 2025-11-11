from __future__ import annotations

import numpy as np
import math
import json
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from common.tracer_monitor import TracerMonitor
import copy
import random
from common.tracers import (
    FrontSameLaneTracer,
    CutOutTracer,  # 1
    FrontSameLaneTracer,
    FrontSlowDownSameLaneTracer,
    BehindSameLaneTracer,
    BehindSpeedUpTracer,  # 2
    FrontDifferentLaneTracer,
    CutInTracer,  # 3 ego lane 0
    FrontDifferentLaneTracer,
    FrontSlowDownDifferentLaneTracer,  # ego lane 0
    SideTracer,
    CutInTracer,  # 4 ego lane 0
    BehindDifferentLaneTracer,
    CutInTracer,  # 5 ego lane 0
    BehindDifferentLaneTracer,
    BehindSpeedUpTracer,  # ego lane 0
    FrontDifferentLaneTracer,
    FrontSlowDownDifferentLaneTracer,  # 6 ego lane 1
    FrontDifferentLaneTracer,
    CutInTracer,  # ego lane 1
    SideTracer,
    CutInTracer,  # 7 ego lane 1
    BehindDifferentLaneTracer,
    CutInTracer,  # 8 ego lane 1
    BehindDifferentLaneTracer,
    BehindSpeedUpTracer,  # ego lane 1
    CutInSideTracer,
    EgoCutInTracer,
    EgoCutInSideTracer,
    EgoCutOutTracer,
)
from common.trace_analyzer import TraceAnalyzer
from common.trace_recorder import TraceRecorder
from highway_env.envs import HighwayEnvFast

Observation = np.ndarray


def norm(value=None, min_value=None, max_value=None, new_min=0, new_max=1):
    """
    Normalize a value from an original range [min_value, max_value] to a new range [new_min, new_max].

    :param value: The value to normalize.
    :param min_value: The minimum value of the original range.
    :param max_value: The maximum value of the original range.
    :param new_min: The minimum value of the new range (default is 0).
    :param new_max: The maximum value of the new range (default is 1).
    :return: The normalized value.
    """
    if min_value == max_value:
        raise ValueError("min_value and max_value cannot be the same")

    return new_min + (value - min_value) * (new_max - new_min) / (max_value - min_value)


class HighwayEnvAdversary(HighwayEnvFast):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        print("=== CustomEnv.__init__ called ===")

        self.tracer = CutInTracer()
        self.tracer_monitor = TracerMonitor(
            [
                EgoCutInSideTracer(),
                EgoCutInTracer(),
                EgoCutOutTracer(),
                CutInSideTracer(),
                CutOutTracer(),
                CutInTracer(),
                FrontSlowDownSameLaneTracer(),
                FrontSlowDownDifferentLaneTracer(),
            ]
        )
        self.trace_analyzer = TraceAnalyzer()
        self.trace_recorder = TraceRecorder(save_folder="trace_record", episode=0)
        self.last_actions = 8  # 8
        self.action_size = 5
        self.all_failure_actions_dict = {}
        self.all_failure_count = {}
        self.current_failure_id = 0
        self.unique_failures_num = len(self.all_failure_count)

        self._step = 0
        super().__init__(config, render_mode)
        # self.trace_recorder.trace_recorder_save_path = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 2,
                "vehicles_count": 0,
                "controlled_vehicles": 2,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": 0,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": -1,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def reset_failure_dict(self):
        self.all_failure_actions_dict = {}
        self.all_failure_count = {}
        self.current_failure_id = 0
        self.unique_failures_num = len(self.all_failure_count)

    def load_failure_dict(self, failure_dict: dict):
        self.all_failure_actions_dict = failure_dict
        self.all_failure_count = {i: 1 for i in range(len(failure_dict))}
        self.current_failure_id = len(failure_dict)
        self.unique_failures_num = len(self.all_failure_count)

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.total_good_drive_reward = 0
        self._crashed = False
        self.train = True
        self.total_adv_reward = 0
        self.true_crash = False
        self.crash = False
        self.num_slowdowns = 0
        self.num_lane_changes = 0
        self.ego_previous_velocities = []
        self.adv_previous_velocities = []
        self.ego_previous_speed = None
        self.ego_previous_lane_index = None
        self.adv_previous_lane_index = None
        self.approach_reward = 0
        self.change_lane_reward = 0
        self._crashed_once = False
        self.follow_reward = 0
        self.tracer.reset()
        self.tracer_monitor.reset()
        self.trace_recorder.reset()
        self.crash_info = None
        self.failure_actions = []
        self.assigned_failure_id = None
        self.unique_failures_num = len(self.all_failure_count)
        self._step += 1

    def _ego_slowdown_detect(self) -> bool:
        """
        Check if the ego vehicle is slowing down, i.e. its speed is lower than the previous speed.
        :return: True if the ego vehicle is slowing down
        """
        try:
            ego_vehicle = self.controlled_vehicles[0]
            speed = ego_vehicle.velocity[0]
            if self.ego_previous_speed is None:
                self.ego_previous_speed = speed
                return False
            else:
                speed_change = speed - self.ego_previous_speed
                self.ego_previous_speed = speed
                return speed_change < -2
        except Exception as e:
            print(f"Error in ego slowdown detect: {e}")
            return False

    def _ego_lane_change_detect(self) -> bool:
        """
        Check if the ego vehicle is changing lane, i.e. its lane index is different from the previous lane index.
        :return: True if the ego vehicle is changing lane
        """
        try:
            ego_vehicle = self.controlled_vehicles[0]
            if self.ego_previous_lane_index is None:
                self.ego_previous_lane_index = ego_vehicle.lane_index
                return False
            else:
                lane_change = ego_vehicle.lane_index != self.ego_previous_lane_index
                self.ego_previous_lane_index = ego_vehicle.lane_index
                return lane_change
        except Exception as e:
            print(f"Error in ego lane change detect: {e}")
            return False

    def _adv_lane_change_detect(self) -> bool:
        """
        Check if the adversary vehicle is changing lane, i.e. its lane index is different from the previous lane index.
        :return: True if the adversary vehicle is changing lane
        """
        try:
            adv_vehicle = self.controlled_vehicles[1]
            if self.adv_previous_lane_index is None:
                self.adv_previous_lane_index = adv_vehicle.lane_index
                return False
            else:
                lane_change = adv_vehicle.lane_index != self.adv_previous_lane_index
                self.adv_previous_lane_index = adv_vehicle.lane_index
                return lane_change
        except Exception as e:
            print(f"Error in adversary lane change detect: {e}")
            return False

    def get_safe_lateral_distance(self, velocity: list) -> float:
        """
        Calculate a safe lateral distance based on the speed of the vehicle.
        The formula is a simplified version of the 2-second rule.
        """
        v_lat = velocity[1]
        speed = np.linalg.norm(velocity)

        deceleration = 5
        a_lat = abs(deceleration * v_lat / speed)
        min_distance = 2
        eps = 0.001
        try:
            safe_distance = (v_lat**2) / ((2 * a_lat) + eps) + min_distance
        except:
            safe_distance = min_distance
        return safe_distance

    def _cosine_similarity(self, v1: list, v2: list) -> float:
        """
        Calculate the cosine similarity between two vectors.
        :param v1: First vector
        :param v2: Second vector
        :return: Cosine similarity value
        """
        if len(v1) < len(v2):
            v1 = v1 + [0] * (len(v2) - len(v1))
        elif len(v2) < len(v1):
            v2 = v2 + [0] * (len(v1) - len(v2))
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _failure_seen_simple(self, last_actions: list) -> bool:
        """
        Check if the last actions are already seen in the failure actions dictionary.
        :param last_actions: List of last actions
        :return: True if the actions are already seen, False otherwise
        """
        failure_id = None
        if len(self.all_failure_actions_dict) > 0:
            for i, actions in enumerate(self.all_failure_actions_dict.values()):
                sim_adv = self._cosine_similarity(actions[0], last_actions[0])
                sim_ego = self._cosine_similarity(actions[1], last_actions[1])

                # print(f"Cosine similarity: {sim}")
                if sim_adv > 0.75 and sim_ego > 0.75 and actions[2] == last_actions[2]:
                    return True, i
        return False, failure_id

    def _failure_seen_cosine(self, last_actions: list) -> tuple[bool, int | None]:
        """
        Check if the last actions are already seen in the failure actions dictionary.

        Instead of returning the first match, this version finds the most similar past failure
        based on cosine similarity.

        :param last_actions: List of last actions [adv_action, ego_action]
        :return: (True, best_match_id) if a similar failure is found, (False, None) otherwise
        """
        best_match_id = None
        best_similarity = 0.0
        threshold = 0.95

        for i, actions in enumerate(self.all_failure_actions_dict.values()):
            sim_adv = (
                actions[0] == last_actions[0]
            )  # self._cosine_similarity(actions[0], last_actions[0])
            sim_ego = (
                actions[1] == last_actions[1]
            )  # self._cosine_similarity(actions[1], last_actions[1])
            sim_pos = self._cosine_similarity(actions[3], last_actions[3])
            # avg_similarity = (sim_adv + sim_ego + sim_pos) / 3
            # sim_threshold = sim_adv > threshold and sim_ego > threshold and sim_pos > threshold
            sim_threshold = sim_adv and sim_ego and sim_pos >= threshold

            if (
                sim_threshold and actions[2] == last_actions[2]
            ):  # and avg_similarity > best_similarity
                # best_similarity = avg_similarity
                best_match_id = i

        if best_match_id is not None:
            return True, best_match_id
        return False, 0

    def _euclidian_distance(self, v1: list, v2: list):
        """
        Compute the Euclidean distance between two vectors.

        Parameters:
            vec1 (array-like): First vector.
            vec2 (array-like): Second vector.

        Returns:
            float: Euclidean distance between vec1 and vec2.
        """

        if len(v1) < len(v2):
            v1 = v1 + [0] * (len(v2) - len(v1))
        elif len(v2) < len(v1):
            v2 = v2 + [0] * (len(v1) - len(v2))
        vec1 = np.array(v1)
        vec2 = np.array(v2)

        return float(np.linalg.norm(vec1 - vec2))

    def _failure_seen(self, last_actions: list) -> tuple[bool, int | None]:
        """
        Check if the last actions are already seen in the failure actions dictionary.

        Instead of returning the first match, this version finds the most similar past failure
        based on cosine similarity.

        :param last_actions: List of last actions [adv_action, ego_action]
        :return: (True, best_match_id) if a similar failure is found, (False, None) otherwise
        """
        best_match_id = None
        best_dist = 100
        avg_dist = 100
        threshold_1 = 0.3  # 0.15
        threshold_2 = 0.2  # 0.25

        for i, actions in enumerate(self.all_failure_actions_dict.values()):
            # sim_adv = self._euclidian_distance(actions[0], last_actions[0])
            # sim_ego = self._euclidian_distance(actions[1], last_actions[1])
            sim_pos = self._euclidian_distance(actions[3], last_actions[3])
            avg_dist = sim_pos  # (sim_adv + sim_ego + sim_pos) / 3
            # sim_threshold = sim_adv > threshold and sim_ego > threshold and sim_pos > threshold
            sim_threshold = sim_pos <= threshold_2

            if sim_threshold and actions[2] == last_actions[2] and avg_dist < best_dist:
                best_dist = avg_dist
                best_match_id = i

        if best_match_id is not None:
            return True, best_match_id, round(best_dist, 2)
        return False, 0, round(avg_dist, 2)

    def _novelty_scaler(self, times_seen: int) -> float:
        """
        Calculate a novelty scaling factor based on the number of times an action has been seen.
        :param times_seen: Number of times the action has been seen
        :return: Novelty scaling factor
        """
        # scale  = 3/(times_seen + 3)
        # if times_seen > 10:
        #      scale = -0.1
        # return scale
        tau = 5.0
        threshold = 50  # 20
        penalty = 1.0
        k = 5.0
        if times_seen <= threshold:
            scale = math.exp(-times_seen / tau)
        else:
            scale = -penalty * math.tanh((times_seen - threshold) / k)
        return scale

    def normilize_last_positions(self, last_poisition_list: list) -> list:
        l_b = np.array([-50, -5, -30])
        u_p = np.array([50, 5, 30])
        norm_pos_list = []
        for pos in last_poisition_list:
            pos = np.array(pos)
            pos_norm = list((pos - l_b) / (u_p - l_b))
            norm_pos_list.extend(pos_norm)
        # v_norm = (v - l) / (u - l)
        norm_pos_list = [round(float(x), 3) for x in norm_pos_list]
        return norm_pos_list

    def _get_failure_reward(self) -> float:
        """
        Calculate the reward for a failure, such as a crash.
        :return: The reward for a failure
        """
        all_frames = copy.deepcopy(self.trace_recorder.all_frames_dict)
        ego_fault, info, metadata = self.trace_analyzer.analyze(
            self.tracer_monitor.tracer_dict, all_frames
        )
        reward = 10
        self.crash_info = info
        if ego_fault:
            reward = 30
            self.true_crash = True
            if len(self.trace_recorder.all_frames_dict) > self.last_actions:
                last_frames = list(self.trace_recorder.all_frames_dict.values())[
                    -self.last_actions :
                ]
            else:
                last_frames = list(self.trace_recorder.all_frames_dict.values())
            last_actions = [
                [
                    float(frame["adv_action"] / self.action_size)
                    for frame in last_frames
                ],
                [
                    float(frame["ego_action"] / self.action_size)
                    for frame in last_frames
                ],
                info,
            ]
            last_positions = [
                [
                    frame["adv_x"] - frame["ego_x"],
                    frame["adv_lane"] - frame["ego_lane"],
                    frame["adv_speed"] - frame["ego_speed"],
                ]
                for frame in last_frames
            ]

            last_positions_norm = self.normilize_last_positions(last_positions)
            last_actions.append(last_positions_norm)

            failure_seen, failure_id, avg_dist = self._failure_seen(last_actions)
            if failure_seen:
                self.all_failure_count[failure_id] += 1
                print(
                    f"Failure seen: {failure_id} count: {self.all_failure_count[failure_id]}"
                )
                times_seen = self.all_failure_count[failure_id]
                # reward = self._novelty_scaler(times_seen)* reward
                self.assigned_failure_id = failure_id
            else:
                failure_id = self.current_failure_id
                self.all_failure_actions_dict[self.current_failure_id] = last_actions
                self.all_failure_count[self.current_failure_id] = 1
                self.assigned_failure_id = self.current_failure_id
                self.current_failure_id += 1
            self.failure_actions.append(last_actions)

            # save_path = f"{self.trace_recorder_save_path}\\{self.current_failure_id}"
            # self.trace_recorder.episode = self._step
            self.trace_recorder.all_frames_dict[self.trace_recorder.current_frame - 1][
                "fail_config"
            ] = last_actions
            self.trace_recorder.all_frames_dict[self.trace_recorder.current_frame - 1][
                "fail_info"
            ] = info
            self.trace_recorder.all_frames_dict[self.trace_recorder.current_frame - 1][
                "metadata"
            ] = metadata
            save_path = f"{self.trace_recorder.save_folder}\\{failure_id}"
            self.trace_recorder.save_trace(save_path=save_path)
            self.tracer_monitor.save(
                f"{save_path}\\tracer_monitor_{avg_dist}_{failure_id}_{self.tracer_monitor.episode}.json"
            )

            trace_analyzer_stats = self.trace_analyzer.all_stats
            with open(
                f"{self.trace_recorder.save_folder}\\trace_analyzer_stats.json", "w"
            ) as f:
                json.dump(trace_analyzer_stats, f, indent=4)

            print("Ego fault crash detected")
        else:
            self.crash = True
            if info == "CutInSideTracer":
                reward = -10
            elif info == "CutInTracer" or info == "FrontSlowDownSameLaneTracer":
                dist = metadata["adv_ego_dist"]
                safe_distance = metadata["safe_distance"]
                reward = (safe_distance - dist) / 5 * -1
            else:
                reward = -1
                print(f"Unknown tracer: {info}")

        return reward

    def _get_safe_distance(self) -> float:
        ego_vehicle = self.controlled_vehicles[0]
        adv_vehicle = self.controlled_vehicles[1]
        if adv_vehicle.velocity[0] > ego_vehicle.velocity[0]:
            lead_vehicle = adv_vehicle
            follow_vehicle = ego_vehicle
        else:
            lead_vehicle = ego_vehicle
            follow_vehicle = adv_vehicle

        # Calculate the distance between the two vehicles
        v_lead = lead_vehicle.velocity[0]
        v_follow = follow_vehicle.velocity[0]
        deceleration = 5
        min_distance = 5
        safe_distance = (
            (v_lead**2 - v_follow**2) / (2 * deceleration) + min_distance + v_follow
        )

        return safe_distance

    def _get_distance(self) -> float:
        ego_vehicle = self.controlled_vehicles[0]
        adv_vehicle = self.controlled_vehicles[1]

        dist = np.linalg.norm(ego_vehicle.position - adv_vehicle.position)
        return dist

    def _get_collision_prob(
        self, safe_distance: float, current_distance: float
    ) -> float:
        """
        Calculate the probability of collision based on the distance between vehicles and the safe distance.
        :param safe_distance: The safe distance between vehicles
        :param current_distance: The current distance between vehicles
        :return: The probability of collision
        """
        if current_distance < safe_distance:
            prob = (safe_distance - current_distance) / safe_distance
            return prob
        else:
            return 0.0

    def _get_adversary_reward_old(self, collision_prob: float = 0) -> float:
        """
        Calculate the reward for the adversary vehicle based on the distance to the ego vehicle and the collision probability.
        :param collision_prob: The probability of collision
        :return: The reward for the adversary vehicle
        """
        min_prob = 0.2

        if collision_prob < min_prob:
            reward = -(min_prob - collision_prob)
        else:
            reward = collision_prob
        ego_vehicle = self.controlled_vehicles[0]
        if not (self._crashed_once):
            if ego_vehicle.crashed:
                # self.done = True
                self._crashed_once = True
                # reward = self._get_failure_reward()
                if self._ego_fault_crash_detect_old():
                    reward = 5  # 10 # 5
                    self.true_crash = True
                else:
                    reward = -1  # -5
                    self.crash = True

        return reward

    def _get_adversary_reward(self, collision_prob: float = 0) -> float:
        """
        Calculate the reward for the adversary vehicle based on the distance to the ego vehicle and the collision probability.
        :param collision_prob: The probability of collision
        :return: The reward for the adversary vehicle
        """
        min_prob = 0.1

        if collision_prob < min_prob:
            reward = -(min_prob - collision_prob)
        else:
            reward = collision_prob  # /2
        ego_vehicle = self.controlled_vehicles[0]
        if not (self._crashed_once):
            if ego_vehicle.crashed:
                # self.done = True
                self._crashed_once = True
                reward = self._get_failure_reward()

        return reward

    def _get_distance_reward(self) -> float:
        distance = self._get_distance()
        min_dist = 7
        if distance < min_dist:
            if distance <= 1:
                distance = 1
            reward = -1 / distance
        else:
            reward = 1 / distance
        return reward

    def _get_collision_reward(self) -> float:
        reward = 0

        ego_vehicle = self.controlled_vehicles[0]
        # if not(self._crashed_once):
        if ego_vehicle.crashed:
            # self.done = True
            self._crashed_once = True
            if self._ego_fault_crash_detect():
                reward = 0  # -5
            else:
                reward = -5  # -10

        return reward

    def _get_negative_collision_reward(self) -> float:
        reward = 0

        ego_vehicle = self.controlled_vehicles[0]
        if ego_vehicle.crashed:
            reward = -10

        return reward

    def _get_approach_reward(self):
        dist_threshold = 30.0
        ego_vehicle = self.controlled_vehicles[0]
        adv_vehicle = self.controlled_vehicles[1]
        adv_lead = adv_vehicle.position[0] - ego_vehicle.position[0]

        if adv_lead > 5 and adv_lead < dist_threshold:
            return 0.1
        elif adv_lead <= 5:
            return -1.0 / 100
        else:
            return -(adv_lead) / dist_threshold / 10

    def _get_lane_change_reward(self, slowdown, lane_change, adv_lane_change) -> float:
        """
        Calculate the reward for lane change.
        :return: The reward for lane change
        """
        collision_reward = 0  # self._get_collision_reward()
        change_reward = 0
        beh_change = False

        if slowdown:
            change_reward += 0.1
            beh_change = True
        elif lane_change:
            change_reward += 0.1
            beh_change = True

        if adv_lane_change and not (beh_change):
            change_reward -= 0.1

        total_reward = change_reward + collision_reward

        return total_reward

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # print(f"updating spec")
        self.tracer.update(
            self.controlled_vehicles[0],
            self.controlled_vehicles[1],
        )
        ##print(f"monitoring step")
        self.tracer_monitor.monitor_step(self.tracer.input_trace)
        # print(f"updating trace")
        self.trace_recorder.update_trace(
            ego_veh=self.controlled_vehicles[0],
            ego_action=action[0],
            adv_veh=self.controlled_vehicles[1],
            adv_action=action[1],
        )

        safe_distance = self._get_safe_distance()
        print(f"Safe distance: {safe_distance}")
        safe_lateral_distance = self.get_safe_lateral_distance(
            self.controlled_vehicles[1].velocity
        )
        print(f"Safe lateral distance: {safe_lateral_distance}")
        current_distace = self._get_distance()
        same_lane = (
            self.controlled_vehicles[0].lane_index
            == self.controlled_vehicles[1].lane_index
        )
        print(f"Same lane: {same_lane}")
        if same_lane:
            collision_prob = self._get_collision_prob(safe_distance, current_distace)
        else:
            collision_prob = self._get_collision_prob(
                safe_lateral_distance, current_distace
            )  # self._get_collision_prob(safe_distance, current_distace)/2 # self._get_collision_prob(safe_lateral_distance, current_distace)#
        print(
            f"Current distance: {current_distace}, Collision probability: {collision_prob}"
        )
        # print(f"Safe distance: {safe_distance}, Current distance: {current_distace}, Collision probability: {collision_prob}")
        slowdown = self._ego_slowdown_detect()
        lane_change = self._ego_lane_change_detect()
        adv_lane_change = self._adv_lane_change_detect()
        if slowdown:
            self.num_slowdowns += 1
        if lane_change:
            self.num_lane_changes += 1
        reward = self._get_adversary_reward(collision_prob)
        print(f"Reward: {reward}")

        self.ego_previous_velocities.append(self.controlled_vehicles[0].velocity)
        self.adv_previous_velocities.append(self.controlled_vehicles[1].velocity)

        approach_reward = (
            self._get_approach_reward() + self._get_negative_collision_reward()
        )
        self.approach_reward += approach_reward
        # change_lane_reward = self._get_lane_change_reward(
        #     slowdown, lane_change, adv_lane_change
        # )
        # self.change_lane_reward += change_lane_reward

        # reward = self._get_distance_reward() + self._get_collision_reward()

        # self.tracer.update(self.controlled_vehicles[0], self.controlled_vehicles[1])

        # cut_in_reward =self.tracer_monitor.monitor_step(self.tracer.input_trace)
        # cut_in_reward = (
        #     self.tracer.evaluate_step(self.tracer.input_trace)
        #     + self._get_collision_reward()
        # )
        # cut_in_reward += self._get_collision_reward()
        # cut_in_reward += self._get_approach_reward()

        # reward += change_lane_reward
        # reward = cut_in_reward

        self.total_adv_reward += reward

        return reward

    def _reward_old(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        safe_distance = self._get_safe_distance()
        current_distace = self._get_distance()
        collision_prob = self._get_collision_prob(safe_distance, current_distace)
        # print(f"Safe distance: {safe_distance}, Current distance: {current_distace}, Collision probability: {collision_prob}")
        slowdown = self._ego_slowdown_detect()
        lane_change = self._ego_lane_change_detect()
        adv_lane_change = self._adv_lane_change_detect()
        if slowdown:
            self.num_slowdowns += 1
        if lane_change:
            self.num_lane_changes += 1
        reward = self._get_adversary_reward(collision_prob)

        self.ego_previous_velocities.append(self.controlled_vehicles[0].velocity)
        self.adv_previous_velocities.append(self.controlled_vehicles[1].velocity)

        approach_reward = (
            self._get_approach_reward() + self._get_negative_collision_reward()
        )
        self.approach_reward += approach_reward
        change_lane_reward = self._get_lane_change_reward(
            slowdown, lane_change, adv_lane_change
        )
        self.change_lane_reward += change_lane_reward

        # reward = self._get_distance_reward() + self._get_collision_reward()

        # self.tracer.update(self.controlled_vehicles[0], self.controlled_vehicles[1])

        # cut_in_reward =self.tracer_monitor.monitor_step(self.tracer.input_trace)
        # cut_in_reward = (
        #     self.tracer.evaluate_step(self.tracer.input_trace)
        #     + self._get_collision_reward()
        # )
        # cut_in_reward += self._get_collision_reward()
        # cut_in_reward += self._get_approach_reward()

        reward += change_lane_reward
        # reward = cut_in_reward

        self.total_adv_reward += reward

        return reward  # cut_in_reward#reward #follow_reward#reward #change_lane_reward# , approach_reward # , reward, reward#,

    def _agent_rewards(self, action: Action, vehicle: Vehicle) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(not (vehicle.crashed)),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
        }

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        i = 0
        # sample a numer from 0 to 1 randomly
        p = np.random.uniform(0, 1)
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )

            if i > 0:
                # vehicle.DEFAULT_TARGET_SPEEDS = np.linspace(20, 40, 3)
                # vehicle.target_speed = 25
                # vehicle.DEFAULT_TARGET_SPEEDS = np.linspace(20, 40, 3)
                # vehicle.position[0] = 240
                vehicle.position[0] += random.randint(-5, 5)
            #    vehicle.speed = 27
            #    vehicle.target_speed = 28
            vehicle = self.action_type.vehicle_class(
                self.road,
                vehicle.position,
                vehicle.heading,
                vehicle.speed,
            )
            vehicle.target_speed = 25
            if i > 0:
                # vehicle.DEFAULT_TARGET_SPEEDS = np.linspace(20, 40, 10)
                vehicle = self.action_type.vehicle_class(
                    self.road,
                    vehicle.position,
                    vehicle.heading,
                    vehicle.speed,
                    target_speeds=np.linspace(20, 33, 5),  # (20, 33, 5)
                )
                # vehicle.target_speed = 25
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            if i > 0:
                vehicle.color = (255, 20, 147)  # rgb color for PINK

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
            i += 1
