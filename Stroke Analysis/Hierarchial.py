import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from matplotlib.lines import Line2D

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'  
data = pd.read_csv(file_path)

# Drop categorical columns for clustering
data_for_clustering = data.drop(columns=['Diagnosis'])  # Replace 'Diagnosis' with your categorical column if different

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Reduce the data to a lower number of dimensions (e.g., 2) using PCA for better visualization and potentially improved clustering
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Hierarchical clustering
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
   
    R = dendrogram(linkage_matrix, **kwargs)
    return R['color_list']

clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering.fit(reduced_data)

# Plot the dendrogram for visualization
plt.figure(figsize=(10, 7))
color_list = plot_dendrogram(clustering, truncate_mode='level', p=3)
plt.title("Dendrogram for the Hierarchical Clustering")
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.ylabel("Distance")

unique_colors = list(set(color for color in color_list if color_list.count(color) > 1))
legend_elements = [Line2D([0], [0], color=color, lw=2, label=f'Cluster {i+1}') for i, color in enumerate(unique_colors)]
plt.legend(handles=legend_elements, loc='upper right')

plt.show()

# Choose the number of clusters based on the dendrogram
num_clusters = 3  #based on dendrogram analysis

# Apply Hierarchical clustering with the chosen number of clusters
hierarchical_cluster = AgglomerativeClustering(n_clusters=num_clusters)
labels = hierarchical_cluster.fit_predict(reduced_data)

# Validate the clustering using the Silhouette Score
silhouette_avg = silhouette_score(reduced_data, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualization of the clusters
plt.figure(figsize=(10, 7))

# Create a scatter plot where each cluster has its own color
# 'unique_labels' will store the unique cluster labels
unique_labels = np.unique(labels)
for label in unique_labels:
    # Select data points belonging to the current label
    clustered_data = reduced_data[labels == label]
    plt.scatter(clustered_data[:, 0], clustered_data[:, 1], label=f'Cluster {label}', marker='o')

plt.title('Cluster Visualization (2D PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Create a legend
plt.legend()

plt.show()
