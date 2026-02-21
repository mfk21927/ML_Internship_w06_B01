# Step 1: Import DBSCAN and other required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


print("Generating non-spherical dataset (moons)...")
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


eps_values = [0.1, 0.3, 0.5, 1.0]
min_samples_values = [3, 5, 10]

print("\n--- Experimenting with DBSCAN Parameters ---")
best_dbscan = None
best_labels = None
# We'll consider the "best" as the one that finds exactly 2 clusters (excluding noise) for this specific dataset
for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # Calculate number of clusters found (ignoring noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"eps={eps}, min_samples={min_samples} -> Clusters: {n_clusters}, Noise points: {n_noise}")
        
        if n_clusters == 2 and best_dbscan is None:
            best_dbscan = db
            best_labels = labels
            best_eps = eps
            best_min_samples = min_samples

# Using the successful parameters for the visualization 
if best_dbscan is None:
    best_eps, best_min_samples = 0.3, 5
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_labels = best_dbscan.fit_predict(X_scaled)

#  Identify noise points and calculate number of clusters found
n_clusters_found = len(set(best_labels)) - (1 if -1 in best_labels else 0)
noise_mask = best_labels == -1
print(f"\nProceeding with visualization using eps={best_eps}, min_samples={best_min_samples}")
print(f"Total clusters found: {n_clusters_found}")
print(f"Total noise points identified: {np.sum(noise_mask)}")

#  Apply K-Means on same data for comparison
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot DBSCAN
scatter_db = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, cmap='viridis', s=50, edgecolor='k')
# Highlight noise points with red 'x'
ax1.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], c='red', marker='x', s=50, label='Noise')
ax1.set_title(f'DBSCAN (eps={best_eps}, min_samples={best_min_samples})')
ax1.legend()

# Plot K-Means
scatter_km = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50, edgecolor='k')
ax2.set_title('K-Means Clustering (K=2)')

plt.suptitle('DBSCAN vs K-Means on Non-Spherical Data', fontsize=16)
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("DOCUMENTATION: DBSCAN vs K-Means")
print("="*50)
print("When to use DBSCAN:")
print("1. Arbitrary Shapes: When clusters are not spherical or have complex, interlocking shapes (like the moons dataset above).")
print("2. Unknown 'K': When you don't know the number of clusters in advance; DBSCAN determines this based on density.")
print("3. Noise Handling: When the dataset contains significant noise or outliers. DBSCAN isolates these as -1 rather than forcing them into a cluster.")
print("\nWhen to use K-Means:")
print("1. Spherical/Globular Clusters: When data groups are roughly circular and equally sized.")
print("2. Large Datasets: K-Means is generally faster and more scalable to very large datasets compared to DBSCAN.")
print("3. Guaranteed Assignments: When every point MUST be assigned to a cluster (no unclassified/noise points allowed).")
print("="*50)