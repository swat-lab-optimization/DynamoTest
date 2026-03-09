import os
import re
import subprocess

# Define arguments


# Call the target script with the two arguments

for fold in range(220, 250):
    video_folder = f"stats\\final_22_oct_uc2_tmp\\rl_2025-10-25-4005-dqn_baseline_defensive_uc2_ga_200_&rl_2.2_no_nov\\run_10\\{fold}"
    print(f"{video_folder}")
    for file in sorted(os.listdir(video_folder)):
        if "scenario" in file:
            match = re.search(r"(\d+)", file)
            recording_episode = int(match.group(1))
            print(f"Recording episode {recording_episode}")

            subprocess.run(
                ["python", "replay_one_trace.py", str(fold), str(recording_episode)]
            )

            # recording_init_file = f"{video_folder}\\scenario_init_episode_{recording_episode}.json"
            # recording_file = f"{video_folder}\\scenario_trace_episode_recording_{recording_episode}.json"
