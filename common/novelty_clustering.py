from sklearn.cluster import AgglomerativeClustering

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def upper_triangular_to_square(upper_tri):
    n = len(upper_tri)  # Number of rows in the upper triangular part
    square_matrix = np.zeros((n, n))  # Initialize square matrix with zeros
    
    # Fill the upper triangular part
    for i in range(n):
        for j in range(i + 1, n):  # Only fill when i < j
            square_matrix[i, j] = upper_tri[i][j]
            square_matrix[j, i] = upper_tri[i][j]  # Mirror the value
    
    return square_matrix


def get_clusters_from_distance_matrix(dist_matrix:np.ndarray, threshold:float = 0.2, linkage:str = "single") -> np.ndarray:

    #if out:
    #    threshold = 5#2
    #else:    
    #    threshold = 0.02

    clustering = AgglomerativeClustering(n_clusters=None, linkage=linkage, metric='precomputed', compute_full_tree=True, distance_threshold=threshold)
    clusters= clustering.fit_predict(dist_matrix)

    cluster_num = 1+np.amax(clusters)

    return cluster_num



def get_distance_matrix(test_list:list, dist_func) -> np.ndarray:
    all_novelty = []
    dist_matrix = np.zeros((len(test_list), len(test_list)))
    for i in range(len(test_list)):
        local_novelty = []
        expected_length = len(test_list)* (len(test_list) - 1) // 2
        #print("Expected length", expected_length)
        for j in range(i + 1, len(test_list)): 
            current1 = test_list[i]  # res.history[gen].pop.get("X")[i[0]]
            current2 = test_list[j]  # res.history[gen].pop.get("X")[i[1]]
            nov = dist_func(current1, current2)

            #print("Novelty", nov)
            local_novelty.append(nov)
            dist_matrix[i, j] = nov
            

    full_matrix = upper_triangular_to_square(dist_matrix)
    return full_matrix



def find_clusters(test_list:list, dist_func, threshold=None, linkage="single", plot=False, save_path=None) -> np.ndarray:
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
    distance_matrix = get_distance_matrix(test_list, dist_func)
    cluster_num = get_clusters_from_distance_matrix(distance_matrix, threshold=threshold, linkage=linkage)
    if plot:
        plot_linkage_matrix(distance_matrix, threshold=threshold, method=linkage, save_path=save_path)

    return cluster_num



def plot_linkage_matrix(distances, method='ward', save_path=None, threshold=0):
    condensed_dist_matrix = squareform(distances)
    linkage_matrix = linkage(condensed_dist_matrix, method=method)
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix,  leaf_rotation=90, leaf_font_size=8, color_threshold=threshold)
    plt.axhline(y=threshold, c='k', linestyle='--', label='Distance Threshold')
    plt.title('Dendrogram of Tests')
    plt.xlabel('Test Names')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)  # Save the plot to the specified path
        plt.close()
    else:
        plt.show()
