from dtaidistance import dtw_ndim
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class BehaviorEvaluator:
    """
    A class to evaluate the behavior of a model based on its predictions and ground truth labels.
    """

    def __init__(self):
        """
        Initializes the BehaviorEvaluator with a model and a data loader.

        Args:
            model: The model to evaluate.
            data_loader: The data loader providing the dataset.
        """

        self.episode_data = []  # {"ego_x": [], "ego_y": [], "adv_x": [], "adv_y": []}
        self.all_data = []  # {"ego_x": [], "ego_y": [], "adv_x": [], "adv_y": []}
        self.episode_num = 0

    def record_step(self, ego_veh, adv_veh):
        """
        Records the behavior of the model at a given step.

        Args:
            ego_veh: The ego vehicle.
            adv_veh: The adversary vehicle.
        """
        # Placeholder for recording logic
        # self.episode_data["ego_x"].append(ego_veh.position[0])
        # self.episode_data["ego_y"].append(ego_veh.position[1])
        # self.episode_data["adv_x"].append(adv_veh.position[0])
        # self.episode_data["adv_y"].append(adv_veh.position[1])
        self.episode_data.append([float(adv_veh.position[0]), adv_veh.lane_index[2]])

    def record_behavior(self):
        """
        Records the behavior of the model.

        Args:
            behavior: The behavior to record.
        """
        # self.all_data["ego_x"].append(self.episode_data["ego_x"])
        # self.all_data["ego_y"].append(self.episode_data["ego_y"])
        # self.all_data["adv_x"].append(self.episode_data["adv_x"])
        # self.all_data["adv_y"].append(self.episode_data["adv_y"])
        self.all_data.append(self.episode_data)
        self.episode_num += 1
        self.episode_data = []

    def evaluate_diversity(self):
        """
        Evaluates the model's behavior on the dataset.

        Returns:
            A dictionary containing evaluation metrics.
        """
        # Placeholder for evaluation logic
        all_div_dtw = []
        all_div_hausdorff = []
        for i in range(len(self.all_data) - 1):
            for j in range(i + 1, len(self.all_data)):
                series1 = np.array(self.all_data[i])
                series2 = np.array(self.all_data[j])
                diversity_dtw = 0  # dtw_ndim.distance(series1, series2)
                diversity_hausdorff = directed_hausdorff(series1, series2)[0]
                all_div_dtw.append(diversity_dtw)
                all_div_hausdorff.append(diversity_hausdorff)
        mean_dtw = np.mean(all_div_dtw)
        mean_hausdorff = np.mean(all_div_hausdorff)
        self.all_data = []
        self.episode_num = 0
        return mean_dtw, mean_hausdorff
