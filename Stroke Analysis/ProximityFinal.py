import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/divya/Desktop/Python projects/Stroke_data.csv'
data = pd.read_csv(file_path)

# Preprocessing
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

preprocessed_data = preprocessor.fit_transform(data)

# Distance Calculations
sample_data = preprocessed_data[:50]  # First 50 rows
euclidean_dist_matrix = euclidean_distances(sample_data)
categorical_data = data[categorical_cols]
encoded_categorical_data = preprocessor.named_transformers_['cat'].fit_transform(categorical_data).toarray()
hamming_dist_matrix = pairwise_distances(encoded_categorical_data[:50], metric='hamming')

# Similarity and Dissimilarity Indices
euclidean_dissimilarity = euclidean_dist_matrix / euclidean_dist_matrix.max()
euclidean_similarity = 1 - euclidean_dissimilarity
hamming_dissimilarity = hamming_dist_matrix
hamming_similarity = 1 - hamming_dissimilarity

# Averages for reporting
avg_euclidean_dissimilarity = np.mean(euclidean_dissimilarity)
avg_euclidean_similarity = np.mean(euclidean_similarity)
avg_hamming_dissimilarity = np.mean(hamming_dissimilarity)
avg_hamming_similarity = np.mean(hamming_similarity)

# Display Euclidean Distance Matrix for the first 10 samples
euclidean_dist_df = pd.DataFrame(euclidean_dist_matrix[:10, :10])
print("Euclidean Distance Matrix (First 10 Samples):\n", euclidean_dist_df)

# Display Hamming Distance Matrix for the first 10 samples
hamming_dist_df = pd.DataFrame(hamming_dist_matrix[:10, :10])
print("\nHamming Distance Matrix (First 10 Samples):\n", hamming_dist_df)

# Creating heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(euclidean_dist_matrix, annot=False, cmap='viridis')
plt.title('Heatmap of Euclidean Distances Among Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(hamming_dist_matrix, annot=False, cmap='viridis')
plt.title('Heatmap of Hamming Distances Among Samples (Categorical Data)')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')
plt.show()

# Print the output
print("Average Euclidean Dissimilarity:", avg_euclidean_dissimilarity)
print("Average Euclidean Similarity:", avg_euclidean_similarity)
print("Average Hamming Dissimilarity:", avg_hamming_dissimilarity)
print("Average Hamming Similarity:", avg_hamming_similarity)
