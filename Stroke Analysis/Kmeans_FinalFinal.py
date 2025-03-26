import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, precision_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)



# Assuming 'Gender_Male' is the feature to be dropped based on previous correlation analysis
data_reduced = data.drop(columns=['Gender_Male'], errors='ignore')

# Prepare the data for clustering (excluding non-numeric and the target variable 'Diagnosis')
data_for_clustering = data_reduced.select_dtypes(include=[np.number])
data_for_clustering = data_for_clustering.drop(columns=['Diagnosis'], errors='ignore')

# Standardize the data before PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Find the optimal number of clusters using the Elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.show()

# Assuming the optimal number of clusters from the elbow chart is 5
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_pca)

# Calculate silhouette score
silhouette_avg = silhouette_score(data_pca, clusters)
print('Silhouette Score:', silhouette_avg)



# Scatter plot of the two principal components colored by cluster label
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('2D PCA of Data Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Create a legend
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

# Add cluster labels to the original data
data_reduced['Cluster'] = clusters

# Calculate mean values of original variables for each cluster
# Select only numeric columns for mean calculation
numeric_cols = data_reduced.select_dtypes(include=[np.number]).columns
cluster_variable_means = data_reduced.groupby('Cluster')[numeric_cols].mean()
print(cluster_variable_means.T)  # Transposed for vertical display

data_reduced['Stroke_Numeric'] = data_reduced['Diagnosis'].map({'Stroke': 1, 'No Stroke': 0})

# Ensure that you are only selecting numeric columns
numeric_cols = data_reduced.select_dtypes(include=[np.number]).columns


# Get the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
print("Cluster Centers:\n", cluster_centers)


 #Get the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])

# Generate a heatmap for the cluster centers
plt.figure(figsize=(8, 4))
sns.heatmap(cluster_centers.T, annot=True, cmap="YlGnBu")
plt.title('Heatmap of Cluster Centers')
plt.xlabel('Cluster')
plt.ylabel('Principal Component')
plt.show()



