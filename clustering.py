# =============================================================================
# COMPLETE CLUSTERING LAB - CORRECTED VERSION
# All comments in English as requested
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# SECTION 1: K-MEANS INTRODUCTION - FRIENDS DATASET
# =============================================================================

print("=" * 60)
print("SECTION 1: K-MEANS INTRODUCTION - FRIENDS DATASET")
print("=" * 60)

# Create a list of friend names and repeat to get 100 friends
friend_names = ['Ala', 'Kuba', 'Tomek', 'Dawid', 'Kasia',
                'Kamila', 'Janek', 'Rysiek', 'Anna', 'Emilia'] * 10

# Build a dictionary with synthetic data
data = {
    'Friend': [f'{friend_names[i]}_{i+1}' for i in range(100)],
    'Sports': np.random.randint(1, 11, 100),   # Random interest in sports (1-10)
    'Movies': np.random.randint(1, 11, 100),   # Random interest in movies (1-10)
    'Music': np.random.randint(1, 11, 100)     # Random interest in music (1-10)
}

# Convert to DataFrame
df_friends = pd.DataFrame(data)
print("\nFirst 5 rows of Friends dataset:")
print(df_friends.head())

# Drop the non-numeric 'Friend' column to get feature matrix X
X_friends = df_friends.drop('Friend', axis=1)
print("\nFeature matrix shape:", X_friends.shape)

# --- ELBOW METHOD ---
# We test cluster counts from 1 to 10 and collect WCSS (Within-Cluster Sum of Squares)
wcss_list = []
K_range = range(1, 11)

for i in K_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_friends)
    wcss_list.append(kmeans.inertia_)  # inertia_ is the WCSS value

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss_list, marker='o', linestyle='--', color='b')
plt.title('Elbow Method - Friends Dataset')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# --- APPLY K-MEANS WITH K=2 ---
kmeans_friends = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
df_friends['Cluster'] = kmeans_friends.fit_predict(X_friends)

# --- VISUALIZE CLUSTERS ---
plt.figure(figsize=(10, 6))
# Scatter plot: Sports vs Music, colored by cluster
plt.scatter(df_friends['Sports'], df_friends['Music'],
            c=df_friends['Cluster'], cmap='viridis', s=100, alpha=0.7)

# Plot cluster centers (centroids)
centers = kmeans_friends.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 2], marker='o', c='red', s=200, alpha=0.8, edgecolors='k', label='Centroids')

plt.xlabel('Sports Interest')
plt.ylabel('Music Interest')
plt.title('K-Means Clustering (K=2) - Friends Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nCluster centers:")
print(pd.DataFrame(centers, columns=['Sports', 'Movies', 'Music']))

# =============================================================================
# SECTION 2: EXERCISE 1 - BASICS WITH make_blobs
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: EXERCISE 1 - K-MEANS BASICS WITH make_blobs")
print("=" * 60)

# Generate synthetic data: 300 samples, 4 true centers, 2 features
X_blobs, y_true = make_blobs(n_samples=300, centers=4, random_state=RANDOM_STATE)

# Plot raw data
plt.figure(figsize=(8, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=50, alpha=0.7, edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Exercise 1: Raw Data Points (make_blobs)")
plt.grid(True, alpha=0.3)
plt.show()

# --- ELBOW METHOD FOR make_blobs ---
wcss_blobs = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE, n_init=10)
    km.fit(X_blobs)
    wcss_blobs.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss_blobs, marker='o', linestyle='--', color='g')
plt.title('Elbow Method - make_blobs Dataset')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# --- COMPARE K=3 AND K=4 (first two suitable values from elbow) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, k in enumerate([3, 4]):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_blobs)

    axes[idx].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    axes[idx].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                      c='red', s=200, marker='X', edgecolors='k', label='Centroids')
    axes[idx].set_title(f'K-Means with K={k}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 3: MUSIC GENRES DATASET SETUP
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: MUSIC GENRES DATASET SETUP")
print("=" * 60)

# NOTE: In the original lab, you upload 'echonest.csv' and 'tracks.csv' manually.
# If files are not present, we generate realistic synthetic music data
# that mimics the FMA (Free Music Archive) structure.

try:
    # Try to load real files if they exist in working directory
    feats_col_list = [0,1,2,3,4,5,6,7,8]
    feats_df = pd.read_csv('echonest.csv', usecols=feats_col_list,
                           low_memory=False, header=2)
    feats_df.rename(columns={'Unnamed: 0': 'track_id'}, inplace=True)

    genre_col_list = [0, 40]
    genre_df = pd.read_csv("tracks.csv", usecols=genre_col_list,
                           low_memory=False, header=1)
    genre_df.rename(columns={'Unnamed: 0': 'track_id'}, inplace=True)

    print("Real data files loaded successfully.")

except FileNotFoundError:
    print("Real CSV files not found. Generating synthetic music data for demonstration...")

    # Generate synthetic tracks with 8 acoustic features
    n_tracks = 8000
    genres = ['Rock', 'Electronic', 'Hip-Hop', 'Jazz', 'Classical',
              'Pop', 'Folk', 'Alternative', 'Experimental', 'Blues',
              'Soul', 'Country', 'Latin', 'Reggae', 'R&B', 'Metal']

    # Create track features with some genre-specific patterns
    track_ids = np.arange(1000, 1000 + n_tracks)
    genre_labels = np.random.choice(genres, n_tracks)

    # Base feature values that differ slightly per genre to create clusters
    base_features = {
        'acousticness': {'Rock': 0.3, 'Electronic': 0.1, 'Hip-Hop': 0.2, 'Jazz': 0.6,
                         'Classical': 0.9, 'Pop': 0.2, 'Folk': 0.8, 'Alternative': 0.4,
                         'Experimental': 0.5, 'Blues': 0.7, 'Soul': 0.5, 'Country': 0.8,
                         'Latin': 0.4, 'Reggae': 0.5, 'R&B': 0.3, 'Metal': 0.2},
        'danceability': {'Rock': 0.5, 'Electronic': 0.8, 'Hip-Hop': 0.9, 'Jazz': 0.6,
                         'Classical': 0.2, 'Pop': 0.8, 'Folk': 0.4, 'Alternative': 0.5,
                         'Experimental': 0.3, 'Blues': 0.5, 'Soul': 0.7, 'Country': 0.6,
                         'Latin': 0.9, 'Reggae': 0.8, 'R&B': 0.8, 'Metal': 0.4},
        'energy': {'Rock': 0.9, 'Electronic': 0.8, 'Hip-Hop': 0.7, 'Jazz': 0.4,
                   'Classical': 0.3, 'Pop': 0.7, 'Folk': 0.3, 'Alternative': 0.7,
                   'Experimental': 0.5, 'Blues': 0.4, 'Soul': 0.6, 'Country': 0.5,
                   'Latin': 0.8, 'Reggae': 0.6, 'R&B': 0.6, 'Metal': 0.95},
        'instrumentalness': {'Rock': 0.2, 'Electronic': 0.7, 'Hip-Hop': 0.1, 'Jazz': 0.6,
                             'Classical': 0.95, 'Pop': 0.05, 'Folk': 0.3, 'Alternative': 0.4,
                             'Experimental': 0.8, 'Blues': 0.3, 'Soul': 0.1, 'Country': 0.2,
                             'Latin': 0.2, 'Reggae': 0.3, 'R&B': 0.1, 'Metal': 0.4},
        'liveness': {'Rock': 0.3, 'Electronic': 0.2, 'Hip-Hop': 0.4, 'Jazz': 0.5,
                     'Classical': 0.4, 'Pop': 0.2, 'Folk': 0.3, 'Alternative': 0.3,
                     'Experimental': 0.3, 'Blues': 0.4, 'Soul': 0.4, 'Country': 0.3,
                     'Latin': 0.5, 'Reggae': 0.5, 'R&B': 0.3, 'Metal': 0.4},
        'speechiness': {'Rock': 0.1, 'Electronic': 0.1, 'Hip-Hop': 0.8, 'Jazz': 0.1,
                        'Classical': 0.05, 'Pop': 0.1, 'Folk': 0.1, 'Alternative': 0.1,
                        'Experimental': 0.2, 'Blues': 0.1, 'Soul': 0.1, 'Country': 0.1,
                        'Latin': 0.15, 'Reggae': 0.15, 'R&B': 0.15, 'Metal': 0.15},
        'tempo': {'Rock': 130, 'Electronic': 128, 'Hip-Hop': 95, 'Jazz': 120,
                  'Classical': 90, 'Pop': 120, 'Folk': 100, 'Alternative': 125,
                  'Experimental': 110, 'Blues': 90, 'Soul': 100, 'Country': 110,
                  'Latin': 130, 'Reggae': 80, 'R&B': 90, 'Metal': 150},
        'valence': {'Rock': 0.6, 'Electronic': 0.6, 'Hip-Hop': 0.7, 'Jazz': 0.5,
                    'Classical': 0.4, 'Pop': 0.8, 'Folk': 0.5, 'Alternative': 0.5,
                    'Experimental': 0.4, 'Blues': 0.3, 'Soul': 0.6, 'Country': 0.5,
                    'Latin': 0.8, 'Reggae': 0.7, 'R&B': 0.6, 'Metal': 0.3}
    }

    # Build DataFrame
    feats_df = pd.DataFrame({'track_id': track_ids})
    for feat, genre_map in base_features.items():
        feats_df[feat] = [np.clip(genre_map[g] + np.random.normal(0, 0.1 if feat != 'tempo' else 15), 0, 1)
                          if feat != 'tempo' else genre_map[g] + np.random.normal(0, 15)
                          for g in genre_labels]
        if feat == 'tempo':
            feats_df[feat] = feats_df[feat].clip(lower=40, upper=250)
        else:
            feats_df[feat] = feats_df[feat].clip(lower=0, upper=1)

    genre_df = pd.DataFrame({'track_id': track_ids, 'genre_top': genre_labels})

    print(f"Synthetic data generated: {n_tracks} tracks across {len(genres)} genres.")

# Clean data: remove NaN and reset index
feats_df.dropna(inplace=True)
feats_df.reset_index(drop=True, inplace=True)

genre_df.dropna(inplace=True)
genre_df.reset_index(drop=True, inplace=True)

# Convert text genres to numeric labels
labels_dict = {value: index for index, value in enumerate(genre_df['genre_top'].unique())}
genre_df['genre_numeric_label'] = genre_df['genre_top'].map(labels_dict)

# Merge features and genres (inner join on track_id)
all_data_df = pd.merge(feats_df, genre_df, on='track_id', how='inner')

print("\nFirst 5 rows of merged music data:")
print(all_data_df.head())

print("\nGenre distribution:")
print(all_data_df['genre_top'].value_counts())

# Extract feature matrix (8 acoustic features) for clustering
feature_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'speechiness', 'tempo', 'valence']
X_music = all_data_df[feature_cols].values
y_music_true = all_data_df['genre_numeric_label'].values
genre_names = all_data_df['genre_top'].values

# =============================================================================
# SECTION 4: EXERCISE 2 - PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
# CORRECTION: We now compare PCA on RAW data vs STANDARDIZED data.
# Raw PCA is misleading because 'tempo' (range ~40-250) dominates over
# features scaled 0-1. StandardScaler fixes this by giving all features
# equal weight (mean=0, std=1) before PCA.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: EXERCISE 2 - PCA ON MUSIC FEATURES (CORRECTED)")
print("=" * 60)

# --- 4A: PCA on RAW (unscaled) data ---
# This is shown for comparison only. It produces a misleading result
# because high-magnitude features dominate the components.
pca_raw = PCA(n_components=8)
X_pca_raw = pca_raw.fit_transform(X_music)

explained_variance_raw = pca_raw.explained_variance_ratio_
cumulative_raw = np.cumsum(explained_variance_raw)

print("\n--- PCA on RAW data (MISLEADING - for comparison only) ---")
for i, var in enumerate(explained_variance_raw):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"Cumulative PC1+PC2 (raw): {cumulative_raw[1]*100:.2f}%")

# --- 4B: PCA on STANDARDIZED data (CORRECT approach) ---
# StandardScaler removes scale differences so PCA finds directions
# of maximum variance based on structure, not on units.
scaler = StandardScaler()
X_music_scaled = scaler.fit_transform(X_music)

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_music_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n--- PCA on STANDARDIZED data (CORRECT) ---")
for i, var in enumerate(explained_variance):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nCumulative variance explained by PC1+PC2 (standardized): {cumulative_variance[1]*100:.2f}%")
print(f"Cumulative variance explained by first 4 PCs (standardized): {cumulative_variance[3]*100:.2f}%")

# Plot explained variance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Raw PCA
axes[0].bar(range(1, 9), explained_variance_raw, alpha=0.7, color='coral', label='Individual')
axes[0].plot(range(1, 9), cumulative_raw, marker='o', color='darkred', label='Cumulative')
axes[0].set_title('PCA on RAW Data (Misleading)')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Standardized PCA
axes[1].bar(range(1, 9), explained_variance, alpha=0.7, color='steelblue', label='Individual')
axes[1].plot(range(1, 9), cumulative_variance, marker='o', color='navy', label='Cumulative')
axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
axes[1].set_title('PCA on STANDARDIZED Data (Correct)')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize data on PC1-PC2 plane (STANDARDIZED), colored by TRUE genre
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_music_true, cmap='tab20', s=20, alpha=0.6)
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
plt.title('Exercise 2: Music Genres on PC1-PC2 Plane (Standardized PCA, True Labels)')
plt.colorbar(scatter, label='Genre Numeric Label')
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# SECTION 5: EXERCISE 3 - K-MEANS ON ACOUSTIC FEATURES
# =============================================================================
# CORRECTION: We visualize clusters using the STANDARDIZED PCA projection
# (X_pca from Section 4B) instead of the raw-data PCA. This gives a fair
# 2D view where all 8 features contributed to the components.
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: EXERCISE 3 - K-MEANS ON MUSIC FEATURES (K=actual genres)")
print("=" * 60)

# Determine number of true genres
n_genres = len(np.unique(y_music_true))
print(f"Number of actual genres in data: {n_genres}")

# Apply K-Means with K = number of actual genres (on standardized features)
kmeans_music = KMeans(n_clusters=n_genres, random_state=RANDOM_STATE, n_init=10)
clusters_music = kmeans_music.fit_predict(X_music_scaled)

# Project both true labels and clusters onto STANDARDIZED PC1-PC2 for comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: True genres on standardized PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_music_true, cmap='tab20', s=20, alpha=0.6)
axes[0].set_title('True Genres (Standardized PCA Projection)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# Right: K-Means clusters on standardized PCA
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_music, cmap='tab20', s=20, alpha=0.6)
axes[1].set_title(f'K-Means Clusters (K={n_genres}) on Standardized PCA')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 6: EXERCISE 4 - STRATEGIES TO IMPROVE CLUSTERING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: EXERCISE 4 - IMPROVING CLUSTERING STRATEGIES")
print("=" * 60)

# --- Strategy A: Reduce number of clusters ---
kmeans_k8 = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
clusters_k8 = kmeans_k8.fit_predict(X_music_scaled)

# --- Strategy B: Reduce features (select most discriminative ones) ---
# We select features that intuitively distinguish genres well
selected_features = ['energy', 'danceability', 'tempo', 'acousticness', 'speechiness']
X_music_selected = all_data_df[selected_features].values

# Standardize selected features too
scaler_selected = StandardScaler()
X_music_selected_scaled = scaler_selected.fit_transform(X_music_selected)

pca_selected = PCA(n_components=2)
X_pca_selected = pca_selected.fit_transform(X_music_selected_scaled)

kmeans_selected = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
clusters_selected = kmeans_selected.fit_predict(X_music_selected_scaled)

# --- Strategy C: Reduce both features and clusters ---
# Already combined above (5 features, 8 clusters)

# --- Strategy D: Standardize data before applying PCA (already done in Section 4) ---
# Here we use the standardized full data from Section 4B for consistency.
clusters_scaled = kmeans_k8.fit_predict(X_music_scaled)  # same as Strategy A but explicit

# Print variance comparison
print(f"\n--- Variance Comparison ---")
print(f"Raw data PC1+PC2 variance: {cumulative_raw[1]*100:.2f}% (dominated by tempo)")
print(f"Standardized PC1+PC2 variance: {cumulative_variance[1]*100:.2f}% (fair contribution from all features)")
print(f"Selected features PC1+PC2 variance: {np.sum(pca_selected.explained_variance_ratio_[:2])*100:.2f}%")

# --- VISUALIZATION COMPARISON ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Standardized data, K = n_genres
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_music, cmap='tab20', s=15, alpha=0.6)
axes[0, 0].set_title(f'A) Standardized 8 Features, K={n_genres}')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Reduced clusters K=8
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_k8, cmap='tab20', s=15, alpha=0.6)
axes[0, 1].set_title('B) Standardized Features, K=8')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Selected features
axes[1, 0].scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], c=clusters_selected, cmap='tab20', s=15, alpha=0.6)
axes[1, 0].set_title('C) Selected 5 Features + Standardized, K=8')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Standardized + PCA (same as B but repeated for layout symmetry)
axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_scaled, cmap='tab20', s=15, alpha=0.6)
axes[1, 1].set_title('D) Standardized + PCA, K=8')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Exercise 4: Clustering Strategy Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Evaluate cluster placement and separability
print("\n--- Strategy Evaluation ---")
print("Standardization effect:")
print(f"  Before scaling - PC1 range: [{X_pca_raw[:,0].min():.2f}, {X_pca_raw[:,0].max():.2f}]")
print(f"  After scaling  - PC1 range: [{X_pca[:,0].min():.2f}, {X_pca[:,0].max():.2f}]")
print("  Standardization puts all features on the same scale (mean=0, std=1),")
print("  preventing high-magnitude features (like tempo) from dominating")
print("  both PCA and distance calculations in K-Means.")

# =============================================================================
# SECTION 7: HIERARCHICAL CLUSTERING - AGGLOMERATIVE METHODS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: HIERARCHICAL CLUSTERING - AGGLOMERATIVE")
print("=" * 60)

# Generate small 2D random dataset for clear dendrogram visualization
np.random.seed(0)
X_hier = np.random.rand(20, 2)

print("Sample data for hierarchical clustering:")
print(X_hier[:5])

# Define distance threshold for cutting the dendrogram
threshold = 0.6

# Agglomerative clustering with distance threshold (automatic cluster count)
clustering_hier = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=threshold,
    linkage='ward',
    compute_distances=True
).fit(X_hier)

# Create linkage matrix for scipy dendrogram
linked = linkage(X_hier, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', labels=[f'P{i+1}' for i in range(20)])
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Cut line (threshold={threshold})')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.title('Dendrogram - Ward Linkage')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Show cluster labels
labels_hier = clustering_hier.labels_
print(f"\nNumber of clusters found with threshold={threshold}: {len(np.unique(labels_hier))}")
print(f"Cluster labels: {labels_hier}")

# Create DataFrame with results
df_hier = pd.DataFrame(X_hier, columns=['x', 'y'])
df_hier['cluster'] = labels_hier
print("\nData with cluster assignments:")
print(df_hier.head(10))

# Visualize clusters in 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_hier[:, 0], X_hier[:, 1], c=labels_hier, cmap='tab10', s=200, edgecolors='k', alpha=0.8)
for i, (x, y) in enumerate(X_hier):
    plt.annotate(f'P{i+1}', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Agglomerative Clustering Results ({len(np.unique(labels_hier))} clusters)')
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# SECTION 8: EXERCISE 5 - CUSTOM DATA WITH HIERARCHICAL CLUSTERING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: EXERCISE 5 - CUSTOM DATA HIERARCHICAL CLUSTERING")
print("=" * 60)

# Create custom dataset with 5 clear blobs
X_custom, _ = make_blobs(n_samples=100, centers=5, cluster_std=1.2,
                         random_state=RANDOM_STATE)

# Decide on 5 clusters
n_custom_clusters = 5
agg_custom = AgglomerativeClustering(n_clusters=n_custom_clusters, linkage='ward')
labels_custom = agg_custom.fit_predict(X_custom)

# Dendrogram for custom data
linked_custom = linkage(X_custom, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linked_custom, truncate_mode='lastp', p=20,
           leaf_rotation=90, leaf_font_size=10)
plt.axhline(y=linked_custom[-(n_custom_clusters-1), 2], color='red',
            linestyle='--', label=f'Cut for K={n_custom_clusters}')
plt.xlabel('Data Points (or clusters)')
plt.ylabel('Distance')
plt.title('Exercise 5: Dendrogram for Custom Data')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Visualize final clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_custom[:, 0], X_custom[:, 1], c=labels_custom, cmap='tab10',
            s=100, alpha=0.7, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Exercise 5: Agglomerative Clustering (K={n_custom_clusters})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Assigned {len(np.unique(labels_custom))} clusters to custom dataset.")

# =============================================================================
# SECTION 9: EXERCISE 6 - HIERARCHICAL CLUSTERING ON MUSIC DATA
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: EXERCISE 6 - HIERARCHICAL CLUSTERING ON MUSIC DATA")
print("=" * 60)

# Create 4 clusters using Agglomerative clustering on acoustic features
# We use the standardized features so tempo does not dominate distance
n_hier_music = 4
agg_music = AgglomerativeClustering(n_clusters=n_hier_music, linkage='ward')
hier_clusters = agg_music.fit_predict(X_music_scaled)

# Add cluster info to DataFrame
all_data_df['hier_cluster'] = hier_clusters

# --- Analysis 1: Which clusters do tracks from a SELECTED genre belong to? ---
selected_genre = 'Hip-Hop'  # Change this to explore other genres
genre_subset = all_data_df[all_data_df['genre_top'] == selected_genre]

print(f"\n--- Analysis for genre: {selected_genre} ---")
print(f"Total tracks: {len(genre_subset)}")
print("Distribution across hierarchical clusters:")
print(genre_subset['hier_cluster'].value_counts().sort_index())

# --- Analysis 2: Reverse analysis - which genres are in each cluster? ---
print(f"\n--- Reverse Analysis: Genre composition per cluster ---")
for cluster_id in range(n_hier_music):
    cluster_subset = all_data_df[all_data_df['hier_cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} (total tracks: {len(cluster_subset)}):")
    genre_counts = cluster_subset['genre_top'].value_counts().head(5)
    for g, count in genre_counts.items():
        pct = 100 * count / len(cluster_subset)
        print(f"  {g}: {count} tracks ({pct:.1f}%)")

# --- Check if genres are evenly distributed or concentrated ---
print("\n--- Concentration Analysis ---")
concentration_results = []
for genre in all_data_df['genre_top'].unique():
    genre_data = all_data_df[all_data_df['genre_top'] == genre]
    cluster_counts = genre_data['hier_cluster'].value_counts()
    max_pct = 100 * cluster_counts.iloc[0] / len(genre_data)
    dominant_cluster = cluster_counts.index[0]
    concentration_results.append({
        'Genre': genre,
        'Dominant_Cluster': dominant_cluster,
        'Concentration_%': max_pct,
        'Total_Tracks': len(genre_data)
    })

conc_df = pd.DataFrame(concentration_results).sort_values('Concentration_%', ascending=False)
print("\nGenre concentration in single cluster (top 10):")
print(conc_df.head(10).to_string(index=False))

# Visualize: Heatmap of Genre vs Cluster
cross_tab = pd.crosstab(all_data_df['genre_top'], all_data_df['hier_cluster'])
cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)  # Normalize by row

plt.figure(figsize=(10, 12))
plt.imshow(cross_tab_norm.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Proportion of genre in cluster')
plt.yticks(range(len(cross_tab_norm.index)), cross_tab_norm.index)
plt.xticks(range(n_hier_music), [f'Cluster {i}' for i in range(n_hier_music)])
plt.xlabel('Hierarchical Cluster')
plt.ylabel('Music Genre')
plt.title('Exercise 6: Genre Distribution Across Clusters (Row-normalized)')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ALL EXERCISES COMPLETED")
print("=" * 60)