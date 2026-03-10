import argparse
import json
import os


def symbolic_event_vector(
    event_dict, event_to_id=None, window_size=10, failure_marker=999
):
    """
    Convert an event dictionary into a symbolic sequence for clustering.
    If fewer than window_size steps exist, failure is marked right after the last real step,
    and the rest of the sequence is padded with zeros (to keep length = window_size + 1).
    """

    # Sort steps numerically
    sorted_steps = sorted(event_dict.keys(), key=lambda x: int(x))
    n_steps = len(sorted_steps)

    # Build mapping if not provided
    if event_to_id is None:
        event_names = list(event_dict[sorted_steps[0]].keys())
        event_to_id = {name: i + 1 for i, name in enumerate(event_names)}

    # Build event sequence (sum of active event IDs per step)
    sequence = []
    for step in sorted_steps[-window_size:]:  # only last window_size steps if longer
        event_sum = sum(
            event_to_id[name] for name, active in event_dict[step].items() if active
        )
        sequence.append(event_sum)

    # If fewer than window_size steps → pad *after* failure position
    if n_steps < window_size:
        # mark failure immediately after last real step
        sequence.append(failure_marker)
        # pad with zeros until length = window_size + 1
        sequence += [0] * (window_size + 1 - len(sequence))
    else:
        # if >= window_size → failure at the end
        sequence.append(failure_marker)

    return sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=str, required=True, help="Path to the folder containing the results of multiple runs.")
    folder = parser.parse_args().results_folder
    for file in os.listdir(folder):
        print(f"Processing {file}...")
        result_path = os.path.join(folder, file)
        if "plot" in file:
            continue
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
                pattern = "tracer_monitor"

                # List all matching files
                matching_files = [
                    f
                    for f in os.listdir(os.path.join(folder_path, fail_folder))
                    if pattern in f
                ]
                failure_file = os.path.join(folder_path, fail_folder, matching_files[0])
                with open(failure_file) as f:
                    data = json.load(f)
                vector = symbolic_event_vector(data)
                fail_data[res_folder].append(vector)

        save_path = os.path.join(result_path, "semantic_failures.json")
        with open(save_path, "w") as f:
            json.dump(fail_data, f, indent=4)
