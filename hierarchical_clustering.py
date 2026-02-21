
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage


print("Loading and scaling Iris dataset...")
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


methods = ['ward', 'single', 'complete', 'average']

plt.figure(figsize=(15, 10))
for i, method in enumerate(methods, 1):
    plt.subplot(2, 2, i)
    
    Z = linkage(X_scaled, method=method)
    
    # Plot dendrogram
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
    plt.title(f'Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()


optimal_n_clusters = 3
print(f"\nTraining Agglomerative Clustering with n_clusters={optimal_n_clusters} (Ward linkage)...")
hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, cmap='plasma', edgecolor='k', s=60)
plt.title('Hierarchical Clustering Visualized in 2D (PCA)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

print("\nComparing with K-Means...")
kmeans = KMeans(n_clusters=optimal_n_clusters, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

hc_silhouette = silhouette_score(X_scaled, hc_labels)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print("-" * 30)
print("Silhouette Score Comparison:")
print("-" * 30)
print(f"Hierarchical Clustering (Ward): {hc_silhouette:.4f}")
print(f"K-Means Clustering:             {kmeans_silhouette:.4f}")

if hc_silhouette > kmeans_silhouette:
    print("\nResult: Hierarchical Clustering achieved a higher silhouette score on this scaled dataset.")
elif hc_silhouette < kmeans_silhouette:
    print("\nResult: K-Means achieved a higher silhouette score on this scaled dataset.")
else:
    print("\nResult: Both algorithms achieved the exact same silhouette score.")