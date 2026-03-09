import json
import os


def extract_test(file):
    test = {}
    with open(file) as f:
        data = json.load(f)

    init_config = data["0"]
    test["ego_x"] = init_config["ego_x"]
    test["ego_y"] = init_config["ego_lane"]
    test["ego_speed"] = init_config["ego_speed"]
    test["ego_heading"] = init_config["ego_heading"]
    test["ego_target_lane"] = init_config["ego_target_lane"]

    test["adv_x"] = init_config["adv_x"]
    test["adv_y"] = init_config["adv_lane"]
    test["adv_speed"] = init_config["adv_speed"]
    test["adv_heading"] = init_config["adv_heading"]
    test["adv_target_lane"] = init_config["adv_target_lane"]

    adv_actions = []
    for step in data:
        adv_actions.append(data[step]["adv_action"])

    test["adv_actions"] = adv_actions

    return test


def extract_fail_config(file):

    fail_config = {}

    with open(file) as f:
        data = json.load(f)

    for key in reversed(sorted(data.keys(), key=int)):
        if "fail_config" in data[key]:
            fail_config = data[key]["fail_config"]

    return fail_config


if __name__ == "__main__":
    base_folder = "stats\\RQ\\RQ1\\uc1\\rl_2025-10-18-4005-dqn_baseline_safe_dist"
    all_tests = {}
    all_fail_configs = {}
    limit = 3000
    for fd in os.listdir(base_folder):
        if "run" in fd:
            current_folder = os.path.join(base_folder, fd)
            all_tests[fd] = {}
            all_fail_configs[fd] = {}
            vehicle_stats_file = os.path.join(current_folder, "adv_vehicle_stats.json")
            with open(vehicle_stats_file) as f:
                data = json.load(f)
            if limit:
                max_fail_limit = data[str(limit)]["failure_num"]
            for test in os.listdir(current_folder):
                if "." not in test and int(test) <= max_fail_limit:
                    test_folder = os.path.join(current_folder, test)
                    all_test_files = [
                        f for f in os.listdir(test_folder) if "scenario_trace" in f
                    ]
                    test_file = all_test_files[0]
                    test_file_path = os.path.join(test_folder, test_file)
                    test_scenario = extract_test(test_file_path)
                    fail_config = extract_fail_config(test_file_path)
                    all_tests[fd][test] = test_scenario
                    all_fail_configs[fd][test] = fail_config
            print(f"Extracted {len(all_tests[fd])} tests from {fd}")

    save_path = os.path.join(base_folder, "extracted_tests.json")
    with open(save_path, "w") as f:
        json.dump(all_tests, f, indent=4)

    fail_save_path = os.path.join(base_folder, "extracted_fail_configs.json")
    with open(fail_save_path, "w") as f:
        json.dump(all_fail_configs, f, indent=4)
    print(f"Saved extracted tests to {save_path}")
