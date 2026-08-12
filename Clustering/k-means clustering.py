import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class ClusteringModel:
    def __init__(self, df, n_components=3, n_clusters=5):
        self.original_df = df.dropna()
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.top_features = []
        self.feature_importance = None
        self.clustered_df = None

    def perform_pca(self):
        scaled_features = self.scaler.fit_transform(self.original_df)
        self.pca.fit(scaled_features)

        importance = np.abs(self.pca.components_).sum(axis=0)
        self.feature_importance = pd.Series(importance, index=self.original_df.columns, name="PCA Importance")

        self.top_features = self.feature_importance.sort_values(ascending=False).head(self.n_components).index.tolist()

        print(f"Top {self.n_components} features based on PCA importance: {self.top_features}")
        print("\nFeature Importance (based on PCA components):")
        print(self.feature_importance.sort_values(ascending=False))

    def perform_clustering(self):
        scaled = self.scaler.fit_transform(self.original_df[self.top_features])
        clusters = self.kmeans.fit_predict(scaled)

        self.clustered_df = self.original_df.copy()
        self.clustered_df['Cluster'] = clusters

    def plot_3d_clusters(self):
        if self.clustered_df is None or len(self.top_features) < 3:
            raise ValueError("Clustering must be performed and top 3 features must be available before plotting.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green', 'yellow', 'orange']

        for cluster in range(self.n_clusters):
            cluster_data = self.clustered_df[self.clustered_df['Cluster'] == cluster]
            ax.scatter(
                cluster_data[self.top_features[0]],
                cluster_data[self.top_features[1]],
                cluster_data[self.top_features[2]],
                label=f'Cluster {cluster}',
                color=colors[cluster % len(colors)]
            )

        ax.set_title('3D Cluster Visualization', fontsize=14)
        ax.set_xlabel(self.top_features[0])
        ax.set_ylabel(self.top_features[1])
        ax.set_zlabel(self.top_features[2])
        ax.legend()

        plt.show()

    def save_results(self, base_filename='clustered_data'):
        if self.clustered_df is not None:
            self.clustered_df.to_csv(f'{base_filename}.csv', index=False)
            print(f"\nClustered data saved to '{base_filename}.csv'.")

            for cluster in range(self.n_clusters):
                df_cluster = self.clustered_df[self.clustered_df['Cluster'] == cluster]
                df_cluster.to_csv(f'c{cluster}.csv', index=False)
                print(f"Cluster {cluster} saved to 'c{cluster}.csv'.")
        else:
            print("No clustered data to save.")


# Usage example:
if __name__ == "__main__":
    df = pd.read_csv('your_dataset.csv')

    clustering_model = ClusteringModel(df, n_components=3, n_clusters=5)
    clustering_model.perform_pca()
    clustering_model.perform_clustering()
    clustering_model.plot_3d_clusters()
    clustering_model.save_results()
