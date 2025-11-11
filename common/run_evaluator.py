import json
import os


class RunEvaluator:
    def __init__(self):
        self.current_run_data = {}
        self.all_run_data = {}
        self.current_run = 0
        self.episode = 0
        self.all_run_genal_stats = {}

    def update_episode(self, stats, episode):
        self.current_run_data[str(self.episode)] = stats
        self.episode = episode

    def update_run(self, behavior_div):
        general_stats = {}
        general_stats["behavior_diversity"] = behavior_div
        general_stats["true_crash_rate"] = self.current_run_data[str(self.episode - 1)][
            "true_crash_rate"
        ]
        general_stats["crash_rate"] = self.current_run_data[str(self.episode - 1)][
            "crash_rate"
        ]
        all_rewards = [
            self.current_run_data[str(i)]["total_reward"]
            for i in range(len(self.current_run_data))
        ]
        general_stats["mean_reward"] = sum(all_rewards) / len(all_rewards)
        self.all_run_genal_stats[f"run{str(self.current_run)}"] = general_stats
        self.all_run_data[f"run{str(self.current_run)}"] = self.current_run_data
        self.current_run += 1
        self.episode = 0
        self.current_run_data = {}

    def save_evaluation(self, folder_name=""):
        all_stats_name = os.path.join(folder_name, "all_stats.json")
        general_stats_name = os.path.join(folder_name, "general_stats.json")
        print(f"Saving all stats to {all_stats_name}")
        with open(all_stats_name, "w") as f:
            json.dump(self.all_run_data, f, indent=4)
        with open(general_stats_name, "w") as f:
            json.dump(self.all_run_genal_stats, f, indent=4)
