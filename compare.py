"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for processing the experimental results
"""

import argparse
import csv
import json
import logging as log
import os
from itertools import combinations

import igraph as ig
import leidenalg
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

from dynasto.common.cliffsDelta import cliffsDelta
from dynasto.common.novelty_clustering import find_clusters, get_distance_matrix


def plot_clusters(
    tests, labels, out_root, save_name, title="t-SNE visualization of clusters"
):
    """
    Visualize clusters using t-SNE based on a precomputed distance matrix (e.g., Levenshtein distances).

    Args:
        distance_matrix (ndarray): Precomputed (n x n) pairwise distance matrix.
        labels (ndarray): Cluster labels (e.g., from HDBSCAN or hierarchical clustering).
        out_root (str): Output directory.
        save_name (str): Filename for the saved figure.
        title (str): Optional plot title.
    """

    # Ensure numpy array and consistent types
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    distance_matrix = get_distance_matrix(tests, fast_levenshtein)
    distance_matrix_norm = MinMaxScaler().fit_transform(distance_matrix)

    # Build color map
    cmap = plt.colormaps.get_cmap("gist_ncar")
    colors = [
        cmap(i / max(1, len(unique_labels) - 1)) for i in range(len(unique_labels))
    ]

    # ---- t-SNE embedding ----
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=20,
        n_iter=1000,
        learning_rate=200,
        random_state=42,
        verbose=1,
        init="random",
    )

    # distance_matrix is (n x n)
    X_tsne = tsne.fit_transform(distance_matrix_norm)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_labels, colors):
        mask = labels == k
        xy = X_tsne[mask]
        label_name = f"Cluster {k}" if k != -1 else "Noise"
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            color=col,
            label=label_name,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
            s=50,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    # ---- Save ----
    os.makedirs(out_root, exist_ok=True)
    tsne_plot_path = os.path.join(out_root, save_name)
    plt.savefig(tsne_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] t-SNE plot saved to: {tsne_plot_path}")


def fast_levenshtein(seq1, seq2):
    # Convert integer sequences to bytes or strings
    s1 = ",".join(map(str, seq1))
    s2 = ",".join(map(str, seq2))
    return Levenshtein.distance(s1, s2)


def levenshtein_distance(seq1, seq2):
    """
    Compute Levenshtein (edit) distance between two sequences of integers.

    Operations allowed:
        - insertion
        - deletion
        - substitution

    Args:
        seq1, seq2: lists or arrays of integers

    Returns:
        int: Levenshtein distance
    """
    len1, len2 = len(seq1), len(seq2)

    # Initialize DP table (size (len1+1) x (len2+1))
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Base cases
    for i in range(len1 + 1):
        dp[i][0] = i  # deletions
    for j in range(len2 + 1):
        dp[0][j] = j  # insertions

    # Fill the table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[len1][len2]


def euclidian_distance(v1: list, v2: list):
    """
    Compute the Euclidean distance between two vectors.

    Parameters:
        vec1 (array-like): First vector.
        vec2 (array-like): Second vector.

    Returns:
        float: Euclidean distance between vec1 and vec2.
    """

    if len(v1) < len(v2):
        v1 = v1 + [0] * (len(v2) - len(v1))
    elif len(v2) < len(v1):
        v2 = v2 + [0] * (len(v1) - len(v2))
    vec1 = np.array(v1)
    vec2 = np.array(v2)

    return float(np.linalg.norm(vec1 - vec2))


def cluster_number(data: list[float]) -> int:
    # transform to np array a list of lists
    failures_A = np.array(data)

    eps = 0.4  # distance threshold for similarity
    min_cluster_size = 3  # 5  # minimum number of points per cluster
    min_samples = 1  # 5      # minimum number of samples in a neighborhood for a point to be considered a core point

    # labels_A = DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(failures_A)
    # labels_A = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=fast_levenshtein,    cluster_selection_epsilon=1.0  ).fit(failures_A).labels_
    # n_clusters_A = len(set(labels_A)) - (1 if -1 in labels_A else 0)

    k = 5  # number of neighbors
    knn_graph = kneighbors_graph(
        failures_A, n_neighbors=k, include_self=False, metric=fast_levenshtein
    )

    # Convert to igraph format
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(edges=list(zip(sources, targets)), directed=False)

    # Optional: weight edges by similarity (1 / distance)
    weights = knn_graph.data
    g.es["weight"] = weights

    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,  # modularity optimization
        weights=g.es["weight"],
    )

    # Get cluster labels
    labels_A = np.array(partition.membership)

    # Number of clusters
    n_clusters_A = len(set(labels_A))

    return n_clusters_A, labels_A


def get_global_clusters(failures_combined):
    """Perform global clustering across all algorithms' failures in a run."""

    # for mcs in [3, 5, 10]:
    #     for ms in [2, 4, 6]:
    #         clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    #         global_labels = clusterer.fit_predict(failures_combined)
    #         n_clusters = len(set(global_labels)) - (1 if -1 in global_labels else 0)
    #         noise_ratio = np.mean(global_labels == -1)
    #         print(f"min_cluster_size={mcs}, min_samples={ms} → clusters={n_clusters}, noise={noise_ratio:.2f}")
    # clusterer = HDBSCAN(min_cluster_size=3, min_samples=3, metric='euclidean')
    # clusterer.fit(failures_combined)
    # global_labels = clusterer.labels_
    eps = 3  # distance threshold for similarity
    min_cluster_size = 2  #  # minimum number of points per cluster
    min_samples = 2  # 5      # minimum number of samples in a neighborhood for a point to be considered a core point

    # labels_A = DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(failures_combined)
    # cluster_selection_epsilon=0.2
    # labels_A = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,metric=fast_levenshtein, cluster_selection_epsilon=3.0  ).fit(failures_combined).labels_
    # n_global_clusters = len(set(labels_A)) - (1 if -1 in labels_A else 0)
    # print(f"Global clusters found: {n_global_clusters}")
    # num_noise = list(labels_A).count(-1)
    # print(f"Noise points: {num_noise}")
    k = 5  # number of neighbors
    knn_graph = kneighbors_graph(
        failures_combined, n_neighbors=k, include_self=False, metric=fast_levenshtein
    )

    # Convert to igraph format
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(edges=list(zip(sources, targets)), directed=False)

    # Optional: weight edges by similarity (1 / distance)
    weights = knn_graph.data
    g.es["weight"] = weights

    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,  # modularity optimization
        weights=g.es["weight"],
    )

    # Get cluster labels
    labels_A = np.array(partition.membership)

    # Number of clusters
    n_clusters_A = len(set(labels_A))

    n_global_clusters = len(set(labels_A)) - (1 if -1 in labels_A else 0)
    print(f"Global clusters found: {n_global_clusters}")
    num_noise = list(labels_A).count(-1)
    print(f"Noise points: {num_noise}")

    # params: eps 0.3, cluster_size 5

    return labels_A


def get_local_clusters(algo_ids, algo_id, global_labels):
    """
    Count how many global clusters a specific algorithm covers.
    Noise points (-1) are treated as unique single-failure clusters.
    """
    # select only failures from this algorithm
    algo_mask = algo_ids == algo_id
    algo_labels = global_labels[algo_mask]

    # distinct clusters (exclude -1)
    covered_clusters = set(algo_labels) - {-1}

    # count noise points (each = unique failure)
    unique_noise = np.sum(algo_labels == -1)

    # total = clusters + unique isolated failures
    total_failures = len(covered_clusters) + unique_noise

    return total_failures, len(covered_clusters)


def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )
    log.info(start_msg)


def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="A tool for generating test cases for autonomous systems",
        epilog="For more information, please visit ",
    )
    parser.add_argument(
        "--stats-path",
        nargs="+",
        help="The source folders of the metadate to analyze",
        required=True,
    )
    parser.add_argument(
        "--stats-names",
        nargs="+",
        help="The names of the corresponding algorithms",
        required=True,
    )
    parser.add_argument(
        "--plot-name",
        help="Name to add to the plots",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "--problem",
        help="Type of the problem to analyze. Available options: [ads, uav]",
        required=False,
        default="ads",
        type=str,
    )
    in_arguments = parser.parse_args()
    return in_arguments


def build_median_table(
    fitness_list, diversity_list, column_names, plot_name, save_dir="stats"
):
    """
    The function `build_median_table` takes in fitness and diversity lists, column names, and a plot
    name, and creates a table with mean fitness and diversity values, as well as p-values and effect
    sizes if the lists have two elements each, and writes the table to a CSV file.

    Args:
      fitness_list: The `fitness_list` parameter is a list of lists, where each inner list represents
    the fitness values for a particular algorithm. Each inner list should contain the fitness values for
    a specific algorithm.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific algorithm. Each inner list should contain the
    diversity values for that algorithm.
      column_names: The parameter "column_names" is a list of names for the columns in the table. It is
    used to label the different algorithms or methods being compared in the table.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or table
    that will be generated. It will be used to create a CSV file with the results of the function.
    """
    columns = ["Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = ["Fitness"]
    for alg in fitness_list:
        row_0.append(round(np.mean(alg), 3))

    row_1 = ["Mean diversity"]
    for alg in diversity_list:
        row_1.append(round(np.mean(alg), 3))

    if (len(fitness_list) == 2) and (len(diversity_list) == 2):
        row_0.append(
            round(
                mannwhitneyu(fitness_list[1], fitness_list[0], alternative="two-sided")[
                    1
                ],
                3,
            )
        )
        row_0.append(round(cliffsDelta(fitness_list[1], fitness_list[0])[0], 3))
        row_0.append(cliffsDelta(fitness_list[1], fitness_list[0])[1])

        row_1.append(
            round(
                mannwhitneyu(
                    diversity_list[0], diversity_list[1], alternative="two-sided"
                )[1],
                3,
            )
        )
        row_1.append(round(cliffsDelta(diversity_list[0], diversity_list[1])[0], 3))
        row_1.append(cliffsDelta(diversity_list[0], diversity_list[1])[1])
        columns.append("p-value")
        columns.append("Effect size")

    rows = [columns, row_0, row_1]

    log.info(f"Writing results to {plot_name}_res.csv")
    save_dir = save_dir + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, plot_name + "_res.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def build_cliff_data(
    fitness_list, diversity_list, column_names, plot_name, save_dir="stats"
):
    """
    The function `build_cliff_data` takes in fitness and diversity lists, column names, and a plot name,
    and writes the calculated p-values and effect sizes to separate CSV files for fitness and diversity.

    Args:
      fitness_list: The `fitness_list` parameter is a list of fitness values for different pairs of
    data. Each element in the list represents the fitness values for a specific pair of data.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific column or feature. Each inner list should contain the
    diversity values for that column or feature.
      column_names: The `column_names` parameter is a list of column names. It represents the names of
    the columns in the data that you want to analyze.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or data file
    that will be generated. It is used to create the output file names by appending
    "_res_p_value_fitness.csv" and "_res_p_value_diversity.csv" respectively.
    """
    title = ["A", "B", "p-value", "Effect size"]
    rows = [title]
    for pair in combinations(range(0, len(fitness_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                fitness_list[pair[0]], fitness_list[pair[1]], alternative="two-sided"
            )[1]
        )
        delta_value = round(
            cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[1]
        pair_values.append(str(delta_value) + ", " + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info("Writing cliff delta data to file: " + plot_name + "_res_p_value.csv")

    save_dir = save_dir  # + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, plot_name + "_res_p_value_failures.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    rows = [title]
    for pair in combinations(range(0, len(diversity_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                diversity_list[pair[0]],
                diversity_list[pair[1]],
                alternative="two-sided",
            )[1]
        )

        delta_value = round(
            cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[1]
        pair_values.append(str(delta_value) + ", " + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info(f"Writing to {plot_name + '_res_p_value_diversity.csv'}")

    with open(
        os.path.join(save_dir, plot_name + "_res_p_value_diversity.csv"),
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def plot_convergence(dfs, stats_names, plot_name, base_path="stats"):
    """
    Function for plotting the convergence of the algorithms
    It takes a list of dataframes and a list of names for the dataframes, and plots the mean and
    standard deviation of the dataframes

    :param dfs: a list of dataframes, each containing the mean and standard deviation of the fitness of
    the population at each generation
    :param stats_names: The names of the algorithms
    """
    fig, ax = plt.subplots()

    plt.xlabel("Number of steps", fontsize=16)
    plt.ylabel("Failures", fontsize=16)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()

    len_df = np.inf
    for i, df in enumerate(dfs):
        cur_len = len(dfs[i]["mean"])
        if cur_len < len_df:
            len_df = cur_len

    for i, df in enumerate(dfs):
        # x = np.arange(0, len(dfs[i]["mean"]))
        x = np.arange(0, len_df)
        plt.plot(x, dfs[i]["mean"][:len_df], label=stats_names[i])
        plt.fill_between(
            x,
            np.array(dfs[i]["mean"][:len_df] - dfs[i]["std"][:len_df]),
            np.array(dfs[i]["mean"][:len_df] + dfs[i]["std"][:len_df]),
            alpha=0.2,
        )
        plt.legend()

    log.info("Saving plot to " + plot_name + "_convergence.png")
    plt.savefig(
        os.path.join(base_path, plot_name + "_convergence.png"), bbox_inches="tight"
    )
    plt.close()


def calculate_test_list_novelty(
    test_list: list,
) -> np.ndarray:
    """
    Calculate the novelty of a test list.

    This function calculates the novelty of a given test list by comparing each pair of tests
    in the list and calculating the novelty score using the `calc_novelty` function.

    Parameters:
    - test_list (list): A list of test objects.
    - problem (str): The problem type. Default is "robot".

    Returns:
    - novelty (float): The average novelty score of the test list.
    """

    all_novelty = []
    for i in range(len(test_list)):
        local_novelty = []
        for j in range(i + 1, len(test_list)):
            current1 = test_list[i]  # res.history[gen].pop.get("X")[i[0]]
            current2 = test_list[j]  # res.history[gen].pop.get("X")[i[1]]

            nov = fast_levenshtein(
                current1, current2
            )  # levenshtein_distance(current1, current2)#euclidian_distance(current1, current2)/4.9 # normalize
            # print("Novelty", nov)
            local_novelty.append(nov)
        if local_novelty:
            # all_novelty.append(sum(local_novelty) / len(local_novelty))
            all_novelty.append(min(local_novelty))  # min distance to others

    return sum(all_novelty) / len(all_novelty) if all_novelty else 0.0


def plot_boxplot(
    data_list, label_list, name, max_range=None, plot_name="", save_dir="boxplots"
):
    """
     Function for plotting the boxplot of the statistics of the algorithms
    It takes a list of lists, a list of labels, a name, and a max range, and plots a boxplot of the data

    :param data_list: a list of lists, each list containing the data for a particular algorithm
    :param label_list: a list of labels, each label corresponding to the data in the data_list
    :param name: the name of the plot
    :param max_range: the maximum value of the y-axis
    """

    fig, ax1 = plt.subplots(figsize=(18, 8))  # figsize=(8, 4)
    ax1.set_xlabel("Algorithm", fontsize=20)
    # ax1.set_xlabel("Rho value", fontsize=20)
    ax1.set_ylabel(name, fontsize=20)

    ax1.tick_params(axis="both", labelsize=18)

    ax1.yaxis.grid(
        True, linestyle="-", which="both", color="darkgray", linewidth=2, alpha=0.5
    )
    if max_range == None:
        max_vals = [max(x) for x in data_list]
        max_range = max(max_vals) + 0.1 * max(max_vals)
        # max_range = 110
        # max_range = max(data_list) + 0.1*max(data_list)
    top = max_range
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.boxplot(data_list, widths=0.45, labels=label_list)

    plt.subplots_adjust(bottom=0.15, left=0.16)

    save_dir = save_dir  # + "_" + plot_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(
        os.path.join(save_dir, plot_name + "_" + name + ".png"), bbox_inches="tight"
    )
    plt.close()


def analyse(stats_path, stats_names, plot_name):
    """
    Main function for building plots comparing the algorithms
    It takes a list of paths to folders containing the results of the tool runs, and a list of names
    of the runs, and it plots the convergence and the boxplots of the fitness and novelty

    :param stats_path: a list of paths to the folders containing the stats files
    :param stats_names: list of strings, names of the runs
    """
    convergence_paths = []
    fail_paths = []
    stats_paths = []
    conv_flag = False
    file_name = "convergence_data.json"
    fails_file_name = "fail_data.json"
    failures = "semantic_failures.json"
    failures_path = []
    base_path = "stats\\plots_rq3\\uc2_final_leiden"
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    for path in stats_path:
        convergence_paths.append(os.path.join(path, file_name))
        fail_paths.append(os.path.join(path, fails_file_name))
        failures_path.append(os.path.join(path, failures))

    fail_lists = []
    diversity_lists = []
    all_cluster_lists = []
    all_failure_lists = []
    h_all_cluster_lists = []
    for i in range(len(fail_paths)):
        print(f"Fail path {fail_paths[i]}")
        with open(fail_paths[i], encoding="utf-8") as f:
            data = json.load(f)
        with open(failures_path[i], encoding="utf-8") as f:
            failure_data = json.load(f)
        fail_list = []
        div_list = []
        all_failure_list = []
        clust_list = []
        h_clust_list = []
        for run in data:
            print(f"{run}")

            fail_list.append(data[run])
            all_failure_list.append(failure_data[run])
            diversity = calculate_test_list_novelty(
                test_list=failure_data[run],
            )

            num_clusters = find_clusters(
                failure_data[run], dist_func=fast_levenshtein, threshold=4
            )
            # print("Diversity", diversity)
            div_list.append(diversity)
            n_clusters, labels = cluster_number(failure_data[run])
            clust_list.append(n_clusters)
            h_clust_list.append(num_clusters)
            # clust_list.append(num_clusters)
        plot_clusters(
            failure_data[list(data.keys())[-1]],
            labels,
            base_path,
            f"leiden_local_{stats_names[i]}.png",
        )
        # cluster_lists.append(clust_list)
        diversity_lists.append(div_list)
        fail_lists.append(fail_list)
        all_failure_lists.append(all_failure_list)
        all_cluster_lists.append(clust_list)
        h_all_cluster_lists.append(h_clust_list)

    num_algorithms = len(fail_lists)
    num_runs = 8  # len(fail_lists[0])

    # initialize cluster lists for each algorithm
    cluster_lists = [[] for _ in range(num_algorithms)]
    cluster_lists_global = [[] for _ in range(num_algorithms)]

    for run_idx in range(num_runs):
        # Collect all algorithms' failures for this run
        run_failures = [all_failure_lists[a][run_idx] for a in range(num_algorithms)]
        run_failures = [np.array(f) for f in run_failures if len(f) > 0]

        # Skip if empty
        if not run_failures:
            for algo_id in range(num_algorithms):
                cluster_lists[algo_id].append(0)
            continue

        # Combine all algorithms' failures into one array
        failures_combined = np.vstack(run_failures)
        algo_ids = np.concatenate(
            [np.full(len(fails), a) for a, fails in enumerate(run_failures)]
        )

        # Global clustering (shared failure space for this run)

        global_labels = get_global_clusters(failures_combined)
        plot_clusters(
            failures_combined, global_labels, base_path, f"leiden_glob_{run_idx}.png"
        )

        # Count how many clusters each algorithm covers
        for algo_id in range(num_algorithms):
            # print(f"Algo id {algo_id}")
            n_local, n_global = get_local_clusters(algo_ids, algo_id, global_labels)
            cluster_lists[algo_id].append(n_local)
            cluster_lists_global[algo_id].append(n_global)
            # print(f"Local clusters {n_local}")
            # print(f"Global clusters {n_global}")

    plot_boxplot(
        fail_lists,
        stats_names,
        "Number of failures",
        plot_name=plot_name,
        save_dir=base_path,
    )
    plot_boxplot(
        diversity_lists,
        stats_names,
        "Diversity",
        plot_name=plot_name,
        save_dir=base_path,
    )
    plot_boxplot(
        all_cluster_lists,
        stats_names,
        "Leiden clustering",
        plot_name=plot_name,
        save_dir=base_path,
    )
    plot_boxplot(
        h_all_cluster_lists,
        stats_names,
        "Hierarchichal clustering",
        plot_name=plot_name,
        save_dir=base_path,
    )

    plot_boxplot(
        cluster_lists,
        stats_names,
        "Number of glob clusters and unique faults",
        plot_name=plot_name,
        save_dir=base_path,
    )
    plot_boxplot(
        cluster_lists_global,
        stats_names,
        "Number of glob hdb clusters",
        plot_name=plot_name,
        save_dir=base_path,
    )
    build_cliff_data(
        fail_lists, diversity_lists, stats_names, plot_name, save_dir=base_path
    )

    # if conv_flag:
    dfs = {}
    for i, file in enumerate(convergence_paths):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        dfs[i] = pd.DataFrame(data=data)
        dfs[i]["mean"] = dfs[i].mean(axis=1)
        dfs[i]["std"] = dfs[i].std(axis=1)
        # take only 2000
        # dfs[i] = dfs[i].iloc[:2000]

    plot_convergence(dfs, stats_names, plot_name + "_fitness", base_path=f"{base_path}")


if __name__ == "__main__":
    arguments = parse_arguments()
    setup_logging("log.txt", False)

    stats_path = arguments.stats_path
    stats_names = arguments.stats_names
    plot_name = arguments.plot_name

    analyse(stats_path, stats_names, plot_name)

"""
python compare.py --stats-path stats\final\rl_2025-08-10-2005-random_baseline_calibrated stats\final\rl_2025-08-10-2005-cmab_baseline_calibrated stats\final\rl_2025-08-10-2005-dqn_baseline_calibrated stats\final\rl_2025-08-11-2000-dqn_matteo_new_threshold  --stats-names rl 
--plot-name "results_2k"


"""
