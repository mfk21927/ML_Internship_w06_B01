# ML_WEEK6_B01
# 🚀 ML-Internship

> **Name:** Muhammad Fahad 
> **Email:** [![Email](https://img.shields.io/badge/Email-mfk21927@gmail.com-red?style=flat-square&logo=gmail&logoColor=white)](mailto:mfk21927@gmail.com) 
> **LinkedIn:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Fahad-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-fahad-087057293) 
> **Start Date:** 20-12-2025 

---

![Internship](https://img.shields.io/badge/Status-Active-blue?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-B01-orange?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.25-orange?logo=scikitlearn&logoColor=white)]
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview
[cite_start]This repository documents my **Week 6 Machine Learning Internship tasks**, focused on **Clustering & Unsupervised Learning**. 
It includes implementations of **K-Means, Hierarchical Clustering, DBSCAN, and Dimensionality Reduction (PCA & t-SNE)**, along with model evaluation, visualizations, and comparative analysis.

---

## 📈 Week 6 Tasks Overview

| Task | Title | Dataset | Status |
| :--- | :--- | :--- | :--- |
| 6.1 | K-Means Clustering | Iris | ✅ Completed |
| 6.2 | Hierarchical Clustering & Dendrograms | Iris | ✅ Completed |
| 6.3 | DBSCAN & Density-Based Clustering | Moons (Non-spherical) | ✅ Completed |
| 6.4 | Dimensionality Reduction (PCA & t-SNE) | MNIST digits | ✅ Completed |

---

## ✅ Task Details

### [cite_start]**Task 6.1: K-Means Clustering** [cite: 2]

- **Dataset:** Iris (`sklearn.datasets.load_iris`)
- [cite_start]**Steps Implemented:** - Imported KMeans from `sklearn.cluster`[cite: 6].
  - [cite_start]Implemented the Elbow Method by calculating inertia for K=2 to 10[cite: 8].
  - [cite_start]Calculated and plotted the Silhouette Score for each K[cite: 10, 11].
  - [cite_start]Reduced dimensionality to 2D using PCA and visualized the clusters[cite: 12, 13].
  - [cite_start]Compared model initialization using `init='random'` vs `init='k-means++'`[cite: 14].
  - [cite_start]Created cluster profile statistics and saved the cluster labels with the original data[cite: 15, 16].

[cite_start]**Files:** `kmeans_clustering.py`, `iris_with_clusters.csv` [cite: 3]

**Visuals:** Elbow Curve & Silhouette Scores 
![Elbow Curve](figures/Figure_1.png) 

K-Means Clusters Visualized in 2D (PCA) 
![PCA K-Means](figures/pca_means.png) 

---

### [cite_start]**Task 6.2: Hierarchical Clustering & Dendrograms** [cite: 17]

- **Dataset:** Iris (`sklearn.datasets.load_iris`) 
- [cite_start]**Steps Implemented:** - Imported `AgglomerativeClustering` and dendrogram/linkage tools from SciPy[cite: 21, 22].
  - [cite_start]Scaled the dataset and created linkage matrices[cite: 23, 24].
  - [cite_start]Plotted dendrograms using different linkage methods: 'ward', 'single', 'complete', and 'average'[cite: 25, 26].
  - [cite_start]Determined the optimal number of clusters by visual inspection of dendrogram heights[cite: 27].
  - [cite_start]Visualized the resulting clusters with PCA and compared the silhouette scores with K-Means results[cite: 29, 30, 31].

[cite_start]**Files:** `hierarchical_clustering.py` [cite: 18]



Hierarchical Clustering Visualized in 2D (PCA)
![PCA Hierarchical](figures/task2_pca.png) 

---

### [cite_start]**Task 6.3: DBSCAN & Density-Based Clustering** [cite: 32]

- [cite_start]**Dataset:** Make Moons (Non-spherical clusters) [cite: 37]
- [cite_start]**Steps Implemented:** - Imported DBSCAN from `sklearn.cluster`[cite: 36].
  - [cite_start]Experimented with various `eps` values [0.1, 0.3, 0.5, 1.0] and `min_samples` [3, 5, 10][cite: 38, 39].
  - [cite_start]Identified noise points with `label=-1`[cite: 40].
  - [cite_start]Visualized clusters with different colors and explicitly highlighted the noise points[cite: 41, 42].
  - [cite_start]Created side-by-side comparison plots showing DBSCAN vs K-Means on the same dataset[cite: 44, 45].
  - [cite_start]Documented scenarios explaining when to use DBSCAN versus K-Means[cite: 46].

[cite_start]**Files:** `dbscan_clustering.py` [cite: 33]

**Visuals:** DBSCAN vs K-Means Comparison
![DBSCAN vs KMeans](figures/dbscanvsKmeans.png) 

---

### [cite_start]**Task 6.4: Dimensionality Reduction (PCA & t-SNE)** [cite: 47]

- [cite_start]**Dataset:** MNIST Digits [cite: 53]
- [cite_start]**Steps Implemented:** - Imported PCA and TSNE from sklearn[cite: 52].
  - [cite_start]Applied PCA to extract 50 components (`n_components=50`)[cite: 54].
  - [cite_start]Plotted the explained variance ratio and the cumulative explained variance[cite: 55, 56].
  - [cite_start]Reduced the dataset to 2D using both PCA and t-SNE, and created comparative scatter plots[cite: 57, 58, 59].
  - [cite_start]Created 3D visualizations for both PCA and t-SNE projections[cite: 60].
  - [cite_start]Trained a classifier on the original features vs the reduced features to compare accuracies and training times[cite: 61, 62].

[cite_start]**Files:** `dimensionality_reduction.py` [cite: 48]


---

## 🧠 ML Projects

- K-Means Clustering (Optimal K & Initialization comparison)
- Hierarchical Clustering (Dendrogram Analysis & Linkage Evaluation)
- Density-Based Clustering (DBSCAN Parameter Tuning & Noise Identification)
- Dimensionality Reduction (PCA Explained Variance & t-SNE 2D/3D Visualization)

---

## 💻 Tech Stack
* **Languages:** Python, Markdown 
* **Libraries:** NumPy, Pandas, Matplotlib, Scikit-Learn, SciPy 
* **Tools:** Git, VS Code

---

## 📜 License
This project is licensed under the MIT License.