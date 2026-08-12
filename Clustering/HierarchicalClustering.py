import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class HierarchicalClustering:
    def __init__(self, n_clusters=3, strategy='divisive'):
        self.n_clusters = n_clusters
        self.strategy = strategy.lower()
        self.clusters = []
        self.labels_ = None
        self.linkage_matrix = []

        if self.strategy not in ['divisive']:
            raise NotImplementedError(f"Strategy '{self.strategy}' not implemented. Only 'divisive' is supported.")

    def fit(self, data: pd.DataFrame):
        if self.strategy == 'divisive':
            return self._fit_divisive(data)
        else:
            raise NotImplementedError(f"Strategy '{self.strategy}' not implemented.")

    def _fit_divisive(self, data: pd.DataFrame):
        self.clusters = [data.copy()]
        cluster_ids = [0]

        while len(self.clusters) < self.n_clusters:
            # Split the largest cluster
            largest_idx = np.argmax([len(cluster) for cluster in self.clusters])
            largest_cluster = self.clusters.pop(largest_idx)

            if len(largest_cluster) < 2:
                self.clusters.append(largest_cluster)
                break

            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(largest_cluster)

            cluster_1 = largest_cluster[labels == 0]
            cluster_2 = largest_cluster[labels == 1]

            self.clusters.append(cluster_1)
            self.clusters.append(cluster_2)

            self.linkage_matrix.append([cluster_ids[largest_idx], len(self.clusters)-2, 0.0, len(cluster_1)])
            self.linkage_matrix.append([cluster_ids[largest_idx], len(self.clusters)-1, 0.0, len(cluster_2)])

            cluster_ids.extend([len(self.clusters)-2, len(self.clusters)-1])

        self.labels_ = np.zeros(len(data), dtype=int)
        for idx, cluster in enumerate(self.clusters):
            self.labels_[cluster.index] = idx

        return self.labels_

    def plot_clusters(self, df: pd.DataFrame, x_col=None, y_col=None):
        if x_col is None or y_col is None:
            x_col, y_col = df.select_dtypes(include=[np.number]).columns[:2]

        plt.figure(figsize=(10, 6))
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            plt.scatter(cluster_data[x_col], cluster_data[y_col], label=f"Cluster {cluster_id + 1}", s=50)

        plt.title(f'{self.strategy.capitalize()} Clustering Results')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.show()

    def save_clusters(self, df: pd.DataFrame, base_filename="cluster_output"):
        full_output_file = f"{base_filename}_full.csv"
        df.to_csv(full_output_file, index=False)
        print(f"Full clustered data saved to '{full_output_file}'")

        for cluster_id in range(self.n_clusters):
            cluster_data = df[df["Cluster"] == cluster_id]
            filename = f"{base_filename}_c{cluster_id}.csv"
            cluster_data.to_csv(filename, index=False)
            print(f"Cluster {cluster_id} saved to '{filename}'")

    def print_summary(self):
        print(f"\n{self.strategy.capitalize()} Clustering Summary (Total Clusters: {self.n_clusters}):")
        for idx, cluster in enumerate(self.clusters):
            print(f" - Cluster {idx}: {len(cluster)} samples")


if __name__ == "__main__":
    df = pd.read_csv("datasetName.csv")
    features = df.select_dtypes(include=[np.number])

    model = HierarchicalClustering(n_clusters=3, strategy='divisive')
    df['Cluster'] = model.fit(features)

    model.plot_clusters(df)
    model.save_clusters(df, base_filename="datasetName")
    model.print_summary()
