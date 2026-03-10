# Extract timestep and total_reward
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results-folder", type=str, required=True, help="Path to the folder containing the results of multiple runs.")
base_folder = parser.parse_args().results_folder
run_folders = os.listdir(base_folder)

metrics = ["total_reward", "crash_rate", "true_crash_rate", "failure_num"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    i = 0
    all_rewards = []
    for run_folder in run_folders:
        if not run_folder.startswith("run_"):  # or "run_10" in run_folde
            continue
        run_folder_path = os.path.join(base_folder, run_folder)
        base_file = os.path.join(run_folder_path, "adv_vehicle_stats.json")
        with open(base_file) as f:
            data = json.load(f)

        timesteps = [int(k) for k in data.keys()]
        total_rewards = [v[metric] for v in data.values()]
        all_rewards.extend(total_rewards)

        window_size = 30
        sliding_avg = np.convolve(
            total_rewards, np.ones(window_size) / window_size, mode="valid"
        )

        plt.plot(
            timesteps[window_size - 1 :],
            sliding_avg,
            marker="o",
            label=f"{metric}_run{i}",
            alpha=0.7,
        )

        plt.ylim(min(all_rewards) - 0.1, max(all_rewards) + 0.1)  #
        plt.title(f"{metric} over time with sliding window average")
        plt.xlabel("Episode")
        plt.ylabel(f"{metric}")
        plt.grid(True)
        plt.legend()
        # plt.show()
        save_path = os.path.join(base_folder, f"{metric}_sliding_avg.png")
        plt.savefig(save_path)
        i += 1
