import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

#  Loading dataset 
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
df = pd.DataFrame(X, columns=iris.feature_names)


inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("Calculating Inertia and Silhouette Scores for K=2 to 10...")
for k in K_range:
   
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Elbow Curve
ax1.plot(K_range, inertias, marker='o', linestyle='--')
ax1.set_title('Elbow Method')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')

# Silhouette Scores
ax2.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='green')
ax2.set_title('Silhouette Scores')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


optimal_k = 3

#applying pca
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
optimal_labels = kmeans_optimal.fit_predict(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=optimal_labels, cmap='viridis', edgecolor='k', s=60)
plt.title('K-Means Clusters Visualized in 2D (PCA)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster Label')
plt.show()


print("\nComparing Initializations...")
kmeans_random = KMeans(n_clusters=optimal_k, init='random', n_init=1, random_state=42)
kmeans_random.fit(X)

kmeans_plus = KMeans(n_clusters=optimal_k, init='k-means++', n_init=1, random_state=42)
kmeans_plus.fit(X)

print(f"Inertia with 'random' initialization: {kmeans_random.inertia_:.2f}")
print(f"Inertia with 'k-means++' initialization: {kmeans_plus.inertia_:.2f}")


df['Cluster'] = optimal_labels
cluster_profiles = df.groupby('Cluster').mean()
print("\nCluster Profiles (Mean values for each feature):")
print(cluster_profiles)


output_filename = "iris_with_clusters.csv"
df.to_csv(output_filename, index=False)
print(f"\nOriginal data with cluster assignments saved to '{output_filename}'.")