import json
import os

if __name__ == "__main__":
    folder = "stats\\RQ\\RQ2\\uc2"
    for file in os.listdir(folder):
        print(f"Processing {file}...")
        result_path = os.path.join(folder, file)
        print(result_path)
        convergence_data = {}
        fail_data = {}
        for res_folder in os.listdir(result_path):
            print(f"  Analyzing {res_folder}...")
            if not res_folder.startswith("run_"):
                continue
            folder_path = os.path.join(result_path, res_folder)
            base_file = os.path.join(folder_path, "adv_vehicle_stats.json")
            convergence_data[res_folder] = {}
            if not os.path.exists(base_file):
                continue
            with open(base_file) as f:
                data = json.load(f)

            timesteps = [int(k) for k in data.keys()]
            total_failures = [v["failure_num"] for v in data.values()]
            for i, timestep in enumerate(timesteps):
                convergence_data[res_folder][timestep] = total_failures[i]
            fail_num = total_failures[-1]
            fail_data[res_folder] = fail_num

        save_path = os.path.join(result_path, "convergence_data.json")
        with open(save_path, "w") as f:
            json.dump(convergence_data, f, indent=4)

        fail_save_path = os.path.join(result_path, "fail_data.json")
        with open(fail_save_path, "w") as f:
            json.dump(fail_data, f, indent=4)
