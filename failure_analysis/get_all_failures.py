import argparse
import json
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=str, required=True, help="Path to the folder containing the results of multiple runs.")
    folder = parser.parse_args().results_folder
    for file in os.listdir(folder):
        print(f"Processing {file}...")
        if "plot" in file:
            continue
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
