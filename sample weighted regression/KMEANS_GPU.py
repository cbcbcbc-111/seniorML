
from matplotlib import pyplot as plt
import numpy as np
import torch

class KMeansGPU:
    def __init__(self, num_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None,
                 device='cuda'):
        """
        KMeans Clustering using GPU (via PyTorch)

        Parameters:
        - n_clusters: int, default=8
            The number of clusters to form.
        - init: {'k-means++', 'random'}, default='k-means++'
            Method for centroid initialization.
        - n_init: int, default=10
            Number of times the algorithm is run with different centroid seeds.
        - max_iter: int, default=300
            Maximum number of iterations.
        - tol: float, default=1e-4
            Tolerance to declare convergence.
        - verbose: int, default=0
            Verbosity mode.
        - random_state: int or None, default=None
            Determines randomness for centroid initialization.
        - device: str, default='cuda'
            Device to run the computation on, 'cuda' for GPU or 'cpu'.
        """
        self.n_clusters = num_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.device = device

    def _init_centroids(self, X, random_state):
        """Initialize centroids using k-means++ or random."""
        n_samples, n_features = X.shape
        centroids = torch.zeros(self.n_clusters, n_features, device=self.device)

        if self.init == 'k-means++':
            # KMeans++ Initialization
            centroids[0] = X[random_state.randint(0, n_samples)]  # Randomly pick the first centroid

            for i in range(1, self.n_clusters):
                # Compute the squared distances to the centroids
                dist_sq = torch.min(torch.cdist(X, centroids[:i]) ** 2, dim=1)[0]
                # Prevent probabilities from becoming too small
                dist_sq += 1e-10  # Adding a small epsilon value to avoid zero distances
                # Calculate probabilities based on the distances
                probs = dist_sq / dist_sq.sum()
                # Avoid cumulative probabilities being too small
                cumulative_probs = probs.cumsum(0)
                r = random_state.rand()
                # Check if cumulative_probs is not empty and if any value is greater than r
                nonzero_indices = torch.nonzero(cumulative_probs > r)
                if nonzero_indices.numel() == 0:
                    # If no valid indices are found, handle it (e.g., pick the last point)
                    centroids[i] = X[-1]  # Choose the last point as a fallback
                else:
                    # Select the index corresponding to the first cumulative probability > r
                    centroids[i] = X[nonzero_indices[0][0]]


        elif self.init == 'random':
            indices = random_state.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]

        return centroids

    def fit(self, X):
        """Fit KMeans on data X using GPU and return the corresponding labels and cluster centers."""
        # Convert input data to torch tensor and move to device (GPU or CPU)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        random_state = np.random.RandomState(self.random_state)

        # Initialize centroids
        centroids = self._init_centroids(X, random_state)

        # For each run (n_init)
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            labels = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)
            prev_centroids = centroids.clone()

            for i in range(self.max_iter):
                # Compute pairwise distances between X and centroids
                dist_matrix = torch.cdist(X, centroids)  # Shape (n_samples, n_clusters)
                labels = torch.argmin(dist_matrix, dim=1)  # Assign each point to the closest centroid

                # Update centroids
                new_centroids = torch.stack(
                    [X[labels == j].mean(0) if (labels == j).sum() > 0 else centroids[j] for j in range(self.n_clusters)]
                )

                # Check for convergence (if centroids don't change)
                if torch.norm(new_centroids - centroids) < self.tol:
                    break
                centroids = new_centroids

            # Compute inertia (sum of squared distances to closest centroid)
            inertia = (dist_matrix.min(dim=1)[0] ** 2).sum()

            # Keep the best solution based on inertia
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        # Store the best results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self.labels_, self.cluster_centers_



def plot_clusters(X, cluster_ids, centroids):
    """
    可视化聚类结果
    """
    plt.figure(figsize=(8, 6))

    # 绘制每个簇的样本点，使用不同颜色
    unique_clusters = np.unique(cluster_ids)
    colors = plt.cm.get_cmap("tab10", len(unique_clusters))

    for cluster_id in unique_clusters:
        cluster_points = X[cluster_ids == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', color=colors(cluster_id))

    # 绘制簇中心
    centroids_cpu = centroids
    plt.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], c='black', marker='x', s=200, label='Centroids')

    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 创建一些示例数据
    X = np.random.rand(1000, 2)  # 1000 个样本，每个样本 10 个特征

    for i in range(4,10):
        # 设置簇数和设备
        num_clusters = i
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 使用 GPU 加速的 KMeans
        kmeans_gpu = KMeansGPU(num_clusters= num_clusters, max_iter=300, random_state=42, device=device)  # 'cuda' 表示使用 GPU
        labels, cluster_centers = kmeans_gpu.fit(X)
        labels = labels.cpu().numpy()
        cluster_centers = cluster_centers.cpu().numpy()

        # 输出聚类结果
        print("Cluster centers:\n", cluster_centers)
        print("Labels:\n", labels)
        # # 调用自定义的kmeans方法
        # cluster_ids, cluster_centers = kmeans(X, num_clusters, device=device)
        #
        # centroids = cluster_centers.cpu().numpy()
        # labels_pred = cluster_ids.cpu().numpy()
        # cluster_labels, count = np.unique(labels_pred, return_counts=True)  # 获取每个簇的标签和样本数
        # # 输出结果
        # print("cluster_labels", cluster_labels)
        # print("count:", count)
        # 绘制聚类结果
        plot_clusters(X, labels, cluster_centers)

