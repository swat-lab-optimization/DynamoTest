import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay
import json
import os
import csv


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(mp4, video_b64.decode("ascii"))
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


class StatRecorder:
    def __init__(self, filepath=".", train=True, experiment_description=None):
        self.filename = os.path.join(filepath, "adv_vehicle_stats.json")
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        if not (train):
            self.filename = self.filename.replace(".json", "_evaluate.json")

        self.csv_filename = self.filename.replace(".json", ".csv")
        self.experiment_description = experiment_description
        if self.experiment_description:
            # Create a text file to store the experiment description
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            with open(os.path.join(filepath, "experiment_description.txt"), "w") as f:
                f.write(self.experiment_description)

        self.crash_list = [0]
        self.true_crash_list = [0]

    def reset(self):
        self.crash_list = [0]
        self.true_crash_list = [0]

    def save_stats(self, episode=0, env=None):
        filename = self.filename

        if env.crash:
            self.crash_list.append(1)
            self.true_crash_list.append(0)
        elif env.true_crash:
            self.crash_list.append(1)
            self.true_crash_list.append(1)
        else:
            self.crash_list.append(0)
            self.true_crash_list.append(0)

        stats = {
            "total_reward": float(env.total_adv_reward),
            "lane_changes": env.num_lane_changes,
            "slowdowns": env.num_slowdowns,
            "crash_rate": sum(self.crash_list) / len(self.crash_list),
            "true_crash_rate": sum(self.true_crash_list) / len(self.true_crash_list),
            "approach_reward": env.approach_reward,
            "change_lane_reward": env.change_lane_reward,
            "follow_reward": env.follow_reward,
            "crash_info": env.crash_info,
            "crashed": env._crashed_once,
            "ego_fault": env.true_crash,
            "failure_counts": list(env.all_failure_count.values()),
            "failure_id": env.assigned_failure_id,
            "failure_num": env.unique_failures_num,
        }

        result = {episode: stats}
        if episode == 0:
            with open(filename, "w") as f:
                json.dump(result, f, indent=4)
            with open(self.csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = list(stats.keys())
                writer.writerow(header)

        else:
            with open(filename, "r") as f:
                data = json.load(f)
            data.update(result)
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            with open(self.csv_filename, mode="a+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(list(stats.values()))
        print(f"Adversarial vehicle stats saved to {filename}")
        return stats


class ResultAnalyzer:
    def __init__(self, filename="stats.json"):
        self.filename = filename
        self.stats = None
        self.load_stats()
