from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def find_optimal_k_elbow(data, min_clusters=2, max_clusters=10, save_path=None):
    distortions = []
    for i in tqdm(range(min_clusters, max_clusters + 1), desc="Finding optimal k"):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.plot(range(min_clusters, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on the elbow graph): "))
    return optimal_k

def find_optimal_k_silhouette(data, min_clusters=2, max_clusters=10, save_path=None):
    silhouette_scores = []
    for i in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on silhouette score): "))
    return optimal_k


def calculate_gap_statistic(data, min_clusters, k):
    reference_datasets = []
    for _ in range(10):  # 生成10个随机数据集作为参考
        random_data = np.random.rand(*data.shape)
        reference_datasets.append(random_data)

    gap_values = []
    for i in range(min_clusters, k + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        log_inertia = np.log(kmeans.inertia_)

        reference_log_inertias = []
        for ref_data in reference_datasets:
            ref_kmeans = KMeans(n_clusters=i)
            ref_kmeans.fit(ref_data)
            reference_log_inertias.append(np.log(ref_kmeans.inertia_))

        gap = np.mean(reference_log_inertias) - log_inertia
        gap_values.append(gap)

    return gap_values


def find_optimal_k_gap(data, min_clusters=2, max_clusters=10, save_path=None):
    gap_values = calculate_gap_statistic(data, min_clusters, max_clusters)

    plt.plot(range(min_clusters, max_clusters + 1), gap_values, marker='o')
    plt.title('Gap Statistic Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap Value')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on gap statistic): "))
    return optimal_k

