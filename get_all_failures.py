import json
import os

import numpy as np

if __name__ == "__main__":
    # @result_path = "stats\\final_22_oct_uc1_temp\\rl_2025-10-22-4005-dqn_baseline_uc1_ga_200_&rl"
    folder = "stats\\RQ\\RQ2\\uc2"
    for file in os.listdir(folder):
        print(f"Processing {file}...")
        result_path = os.path.join(folder, file)
        print(result_path)
        convergence_data = {}
        fail_data = {}
        for res_folder in os.listdir(result_path):
            if not res_folder.startswith("run_"):
                continue
            print(f"  Analyzing {res_folder}...")
            fail_data[res_folder] = []
            folder_path = os.path.join(result_path, res_folder)
            for fail_folder in os.listdir(folder_path):
                if "." in fail_folder:
                    continue
                pattern = "scenario_trace_episode_recording_"

                # List all matching files
                matching_files = [
                    f
                    for f in os.listdir(os.path.join(folder_path, fail_folder))
                    if pattern in f
                ]
                failure_file = os.path.join(folder_path, fail_folder, matching_files[0])
                with open(failure_file) as f:
                    data = json.load(f)
                    for key in reversed(sorted(data.keys(), key=int)):
                        if "fail_config" in data[key]:
                            fail_config = data[key]["fail_config"]

                            if len(fail_config) >= 4:
                                vector = np.array(fail_config[3])
                                vector_size = len(vector)
                                zero_to_pad = 24 - vector_size

                                vector = np.pad(
                                    vector, (0, max(0, zero_to_pad)), "constant"
                                )
                                fail_data[res_folder].append(vector.tolist())
                            break

        save_path = os.path.join(result_path, "failures.json")
        with open(save_path, "w") as f:
            json.dump(fail_data, f, indent=4)
