
from highway_env.vehicle.controller import ControlledVehicle
import json
import os

class TraceRecorder:
    def __init__(self, save_folder: str, episode: int = 0):
        self.save_folder = save_folder
        self.all_frames_dict= {}
        self.current_frame_dict = {}
        self.current_frame = 0
        self.episode = episode  # Not used in this class, but can be useful for tracking episodes if needed


    def update_trace(self, ego_veh: ControlledVehicle, ego_action:int,  adv_veh: ControlledVehicle, adv_action:int):
        """
        Update the video trace with the given vehicles' information.
        """
        ego_target_lane =  list(ego_veh.target_lane_index)
        ego_target_lane[2] = int(ego_target_lane[2])
        adv_target_lane =  list(adv_veh.target_lane_index)
        adv_target_lane[2] = int(adv_target_lane[2])

        self.current_frame_dict["ego_x"] = float(ego_veh.position[0])
        self.current_frame_dict["ego_lane"] = float(ego_veh.position[1])
        self.current_frame_dict["ego_action"] = (ego_action)
        self.current_frame_dict["ego_speed"] = float(ego_veh.speed)
        self.current_frame_dict["ego_acceleration"] = float(ego_veh.action["acceleration"])
        self.current_frame_dict["ego_heading"] = float(ego_veh.heading)
        self.current_frame_dict["ego_target_lane"] = ego_target_lane
        self.current_frame_dict["adv_x"] = float(adv_veh.position[0])
        self.current_frame_dict["adv_lane"] = float(adv_veh.position[1])
        self.current_frame_dict["adv_action"] = int((adv_action))
        self.current_frame_dict["adv_speed"] = float(adv_veh.speed)
        self.current_frame_dict["adv_acceleration"] = float(adv_veh.action["acceleration"])
        self.current_frame_dict["adv_heading"] = float(adv_veh.heading)
        self.current_frame_dict["adv_target_lane"] = adv_target_lane
        self.current_frame_dict["adv_ego_distance"] = float(adv_veh.position[0] - ego_veh.position[0])
        self.all_frames_dict[self.current_frame] = self.current_frame_dict.copy()  # Store a copy of the current frame data
        self.current_frame += 1


    def save_trace(self, save_path=None):
        """
        Save the video trace to the specified folder.
        """

        
        if save_path:
            save_folder = save_path
        else:
            save_folder = self.save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        trace_file = os.path.join(save_folder, f"scenario_trace_episode_recording_{self.episode}.json")
        with open(trace_file, "w") as f:
            json.dump(self.all_frames_dict, f, indent=4)
        
        print(f"Video trace saved to {trace_file}")
        #self.reset()  # Reset after saving to start a new trace for the next episode   

    def save_trace_init(self, ego: ControlledVehicle, adv: ControlledVehicle):
        """
        Save the initial scenario information to a file.
        """
        init_config = {
            "ego_veh": {
                "position": list(ego.position),
                "speed": list(ego.velocity),
                "heading": float(ego.heading),
                "target_lane_index": ego.target_lane_index

            },
            "adv_veh": {
                "position": list(adv.position),
                "speed": list(adv.velocity),
                "heading": float(adv.heading),
                "target_lane_index": adv.target_lane_index
            }
        }

        scenario_file = os.path.join(self.save_folder, f"scenario_init_episode_{self.episode}.json")
        with open(scenario_file, "w") as f:
            json.dump(init_config, f, indent=4)

        print(f"Scenario initialization saved to {scenario_file}")

    def reset(self):
        """
        Reset the current frame and episode.
        """
        self.all_frames_dict= {}
        self.current_frame_dict = {}
        self.current_frame = 0
        self.episode += 1
