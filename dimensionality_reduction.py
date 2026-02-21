import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time



# We load the dataset and take a subset (10,000 samples) to ensure t-SNE runs in a reasonable time
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X[:10000] / 255.0  # Normalize pixel values
y = y[:10000]


print("Applying PCA with 50 components...")
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X)


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.bar(range(1, 51), pca_50.explained_variance_ratio_, alpha=0.7)
plt.title('Explained Variance Ratio per Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')


plt.subplot(1, 2, 2)
plt.plot(range(1, 51), np.cumsum(pca_50.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Ratio')

plt.tight_layout()
plt.show()


print("\nReducing to 2D with PCA and t-SNE...")
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

tsne_2 = TSNE(n_components=2, random_state=42)
X_tsne_2 = tsne_2.fit_transform(X)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = ax1.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y.astype(int), cmap='tab10', alpha=0.6, s=10)
ax1.set_title('PCA 2D Projection')
fig.colorbar(scatter1, ax=ax1, label='Digit')

scatter2 = ax2.scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=y.astype(int), cmap='tab10', alpha=0.6, s=10)
ax2.set_title('t-SNE 2D Projection')
fig.colorbar(scatter2, ax=ax2, label='Digit')

plt.suptitle('PCA vs t-SNE for Visualization', fontsize=16)
plt.show()


print("\nCreating 3D projections...")
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X)

tsne_3 = TSNE(n_components=3, random_state=42)
X_tsne_3 = tsne_3.fit_transform(X)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y.astype(int), cmap='tab10', s=10, alpha=0.6)
ax1.set_title('PCA 3D Projection')

ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X_tsne_3[:, 0], X_tsne_3[:, 1], X_tsne_3[:, 2], c=y.astype(int), cmap='tab10', s=10, alpha=0.6)
ax2.set_title('t-SNE 3D Projection')

plt.show()


print("\n--- Classification Performance Comparison ---")
# Split data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca = train_test_split(X_pca_50, test_size=0.2, random_state=42)

clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on original features (784 dimensions)
start_time = time.time()
clf_orig.fit(X_train_orig, y_train)
orig_train_time = time.time() - start_time
orig_preds = clf_orig.predict(X_test_orig)
orig_accuracy = accuracy_score(y_test, orig_preds)

# Train on PCA reduced features (50 dimensions)
start_time = time.time()
clf_pca.fit(X_train_pca, y_train)
pca_train_time = time.time() - start_time
pca_preds = clf_pca.predict(X_test_pca)
pca_accuracy = accuracy_score(y_test, pca_preds)


print(f"Original Data (784 features):")
print(f"  - Accuracy:      {orig_accuracy * 100:.2f}%")
print(f"  - Training Time: {orig_train_time:.4f} seconds")

print(f"\nPCA Reduced Data (50 features):")
print(f"  - Accuracy:      {pca_accuracy * 100:.2f}%")
print(f"  - Training Time: {pca_train_time:.4f} seconds")
print("\nConclusion: PCA significantly reduces training time while often maintaining a highly comparable accuracy.")