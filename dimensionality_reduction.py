"""
================================================================================
DIMENSIONALITY REDUCTION LAB - COMPLETE PYTHON SCRIPT
================================================================================
This script covers:
  1. PCA on Iris Dataset (Exercise 1)
  2. PCA on Breast Cancer Dataset (Exercise 2)
  3. PCA on Penguins Dataset (Exercise 3)
  4. PCA on AMP Data placeholder (Exercise 4)
  5. Kernel PCA experiments (make_moons, Iris, Cancer, AMP, MNIST)
  6. Advanced comparison: PCA vs t-SNE vs UMAP (The Great Showdown)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import load_breast_cancer, fetch_openml, make_moons
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

# UMAP (install via: pip install umap-learn)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. UMAP section will be skipped.")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ================================================================================
# SECTION 0: HELPER FUNCTIONS
# ================================================================================

def plot_explained_variance(pca_model, title="Explained Variance"):
    """
    Plot cumulative explained variance vs number of principal components.
    Also prints variance captured by each component.
    """
    evr = pca_model.explained_variance_ratio_
    cumsum = np.cumsum(evr)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(evr)+1), evr, alpha=0.6, label='Individual')
    plt.plot(range(1, len(cumsum)+1), cumsum, 'ro-', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    for i, (r, c) in enumerate(zip(evr, cumsum), 1):
        print(f"PC{i}: {r:.4f} (cumulative: {c:.4f})")
    return cumsum


def find_n_components_for_variance(pca_model, thresholds=[0.99, 0.95, 0.90, 0.85]):
    """
    Find the number of components needed to reach specified variance thresholds.
    """
    cumsum = np.cumsum(pca_model.explained_variance_ratio_)
    results = {}
    for t in thresholds:
        n = np.argmax(cumsum >= t) + 1
        results[t] = n
        print(f"  {int(t*100)}% variance -> {n} components")
    return results


def plot_2d_projection(x_proj, y_labels, title="2D Projection", palette="Set1"):
    """
    Generic 2D scatter plot for dimensionality reduction results.
    """
    plt.figure(figsize=(8, 6))
    if y_labels.dtype == object or len(np.unique(y_labels)) < 10:
        sns.scatterplot(x=x_proj[:, 0], y=x_proj[:, 1], hue=y_labels, palette=palette, alpha=0.7, s=60)
    else:
        scatter = plt.scatter(x_proj[:, 0], x_proj[:, 1], c=y_labels, cmap='Spectral', alpha=0.7, s=20)
        plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class")
    plt.show()


def plot_comparison(data, labels, title_suffix="Dataset", n_neighbors=15, min_dist=0.1, perplexity=30):
    """
    Compare PCA, t-SNE, and UMAP in a 1x3 subplot.
    Measures and reports computation time for each method.
    """
    models = [("PCA", PCA(n_components=2))]

    # t-SNE
    models.append(("t-SNE", TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)))

    # UMAP (if available)
    if UMAP_AVAILABLE:
        models.append(("UMAP", umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)))

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        print(f"Running {name}...")
        start = time.time()
        proj = model.fit_transform(data)
        duration = time.time() - start

        if labels.dtype == object:
            # Convert string labels to numeric for coloring
            unique = np.unique(labels)
            label_map = {u: i for i, u in enumerate(unique)}
            c = np.array([label_map[l] for l in labels])
        else:
            c = labels

        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=c, cmap='Spectral', s=20, alpha=0.7)
        ax.set_title(f"{name}\n({duration:.2f}s)")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {name} completed in {duration:.2f} seconds")

    plt.suptitle(f"Dimensionality Reduction Comparison: {title_suffix}")
    plt.tight_layout()
    plt.show()


# ================================================================================
# SECTION 1: PCA ON IRIS DATASET (Exercise 1)
# ================================================================================

def exercise_1_iris():
    """
    Exercise 1: Principal Component Analysis on the Iris dataset.
    Steps:
      - Load data
      - Visualize pairwise correlations
      - Standardize features
      - Apply PCA (2 components)
      - Visualize 2D projection
      - Report explained variance
    """
    print("\n" + "="*60)
    print("EXERCISE 1: PCA ON IRIS DATASET")
    print("="*60)

    # Load dataset
    df = sns.load_dataset('iris')
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Pairplot to see correlations
    print("\nGenerating pairplot (correlations between features)...")
    sns.pairplot(df, hue='species', palette='Set1')
    plt.suptitle("Iris Pairplot (Original Features)", y=1.02)
    plt.show()

    # Calculate correlation matrix
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    corr = df[features].corr()
    print("\nCorrelation matrix:")
    print(corr)
    print("\nStrong correlations (|r| > 0.8):")
    strong = [(i, j, corr.loc[i, j]) for i in features for j in features
              if i < j and abs(corr.loc[i, j]) > 0.8]
    for i, j, v in strong:
        print(f"  {i} vs {j}: {v:.3f}")

    # Prepare features and labels
    X = df[features].values
    y = df['species'].values

    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    df_std = pd.DataFrame(X_std, columns=features)
    df_std['species'] = y

    print("\nGenerating pairplot for STANDARDIZED data...")
    sns.pairplot(df_std, hue='species', palette='Set1')
    plt.suptitle("Iris Pairplot (Standardized)", y=1.02)
    plt.show()

    # PCA projection to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['species'] = y
    print("\nPCA result (first 5 rows):")
    print(df_pca.head())

    # Visualize 2D projection
    plot_2d_projection(X_pca, y, title="PCA Projection of Iris Dataset", palette=['red', 'green', 'blue'])

    # Explained variance
    print("\nExplained variance ratio:")
    evr = pca.explained_variance_ratio_
    print(f"  PC1: {evr[0]:.4f} ({evr[0]*100:.2f}%)")
    print(f"  PC2: {evr[1]:.4f} ({evr[1]*100:.2f}%)")
    print(f"  Total: {sum(evr):.4f} ({sum(evr)*100:.2f}%)")


# ================================================================================
# SECTION 2: PCA ON BREAST CANCER DATASET (Exercise 2)
# ================================================================================

def exercise_2_breast_cancer():
    """
    Exercise 2: PCA on Breast Cancer Wisconsin dataset.
    Steps:
      - Load and convert to DataFrame
      - Rename labels to 'benign' and 'malignant'
      - Visualize correlations
      - Standardize and apply PCA
      - Plot explained variance
      - Use RFE to split features into top 15 and bottom 15
      - Compare explained variance for: all features, top 15, bottom 15
    """
    print("\n" + "="*60)
    print("EXERCISE 2: PCA ON BREAST CANCER DATASET")
    print("="*60)

    # Load data
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Rename labels for readability
    target_names = {0: 'malignant', 1: 'benign'}
    df['diagnosis'] = df['target'].map(target_names)

    print(f"Dataset shape: {df.shape}")
    print(df['diagnosis'].value_counts())

    # Feature correlation heatmap (first 10 features for readability)
    features = list(data.feature_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[features[:10]].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Breast Cancer: Feature Correlations (First 10)")
    plt.show()

    # Standardize
    X = df[features].values
    y = df['diagnosis'].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA on all 30 features
    pca_all = PCA()
    pca_all.fit(X_std)
    print("\n--- All 30 Features ---")
    cumsum_all = plot_explained_variance(pca_all, "Explained Variance (All Features)")
    find_n_components_for_variance(pca_all)

    # Visualize 2D
    X_pca_2d = PCA(n_components=2).fit_transform(X_std)
    plot_2d_projection(X_pca_2d, y, title="PCA 2D: Breast Cancer (All Features)", palette=['red', 'green'])

    # RFE: Split into top 15 and bottom 15 features
    print("\n--- Recursive Feature Elimination (RFE) ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=clf, n_features_to_select=15, step=1)
    rfe.fit(X_std, df['target'].values)

    top_15_mask = rfe.support_
    bottom_15_mask = ~rfe.support_

    top_features = [f for f, m in zip(features, top_15_mask) if m]
    bottom_features = [f for f, m in zip(features, bottom_15_mask) if m]

    print(f"Top 15 features (by RFE): {top_features}")
    print(f"Bottom 15 features (by RFE): {bottom_features}")

    # PCA on top 15
    X_top = X_std[:, top_15_mask]
    pca_top = PCA()
    pca_top.fit(X_top)
    print("\n--- Top 15 Features ---")
    cumsum_top = plot_explained_variance(pca_top, "Explained Variance (Top 15 RFE Features)")

    # PCA on bottom 15
    X_bottom = X_std[:, bottom_15_mask]
    pca_bottom = PCA()
    pca_bottom.fit(X_bottom)
    print("\n--- Bottom 15 Features ---")
    cumsum_bottom = plot_explained_variance(pca_bottom, "Explained Variance (Bottom 15 RFE Features)")

    # Comparison plot: cumulative explained variance for all 3 cases
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum_all)+1), cumsum_all, 'o-', label='All 30 features', linewidth=2)
    plt.plot(range(1, len(cumsum_top)+1), cumsum_top, 's-', label='Top 15 (RFE)', linewidth=2)
    plt.plot(range(1, len(cumsum_bottom)+1), cumsum_bottom, '^-', label='Bottom 15 (RFE)', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Comparison: All vs Top 15 vs Bottom 15 Features')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nInterpretation:")
    print("  - Top 15 features (ranked by RFE) capture variance more efficiently.")
    print("  - Bottom 15 features require more components to reach the same variance.")
    print("  - This shows that feature importance directly affects PCA efficiency.")


# ================================================================================
# SECTION 3: PCA ON PENGUINS DATASET (Exercise 3)
# ================================================================================

def exercise_3_penguins():
    """
    Exercise 3: PCA on Palmer Penguins dataset.
    Steps:
      - Download and load penguins_size.csv
      - Clean data (remove NaNs)
      - Create numerical labels from species
      - Standardize data and apply PCA
      - Check explained variance and component formulas
      - Apply PCA to NON-standardized data
      - Visualize standardized vs non-standardized in 2D and 3D
      - Test different scaler (RobustScaler) and compare
    """
    print("\n" + "="*60)
    print("EXERCISE 3: PCA ON PENGUINS DATASET")
    print("="*60)

    url = "https://raw.githubusercontent.com/remijul/dataset/master/penguins_size.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Failed to download from URL: {e}")
        # Fallback: try to load from seaborn if available
        try:
            df = sns.load_dataset('penguins')
            df = df.rename(columns={
                'bill_length_mm': 'culmen_length_mm',
                'bill_depth_mm': 'culmen_depth_mm'
            })
            print("Loaded penguins from seaborn as fallback.")
        except Exception:
            print("Could not load penguins dataset. Skipping Exercise 3.")
            return

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Clean data
    df = df.dropna()

    # Numerical labels for species
    species_unique = df['species'].unique()
    species_map = {s: i for i, s in enumerate(species_unique)}
    df['species_label'] = df['species'].map(species_map)
    print(f"\nSpecies mapping: {species_map}")

    # Select features
    penguin_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Ensure columns exist (handle naming variations)
    available_cols = [c for c in penguin_features if c in df.columns]
    if len(available_cols) < 4:
        print(f"Warning: only found columns {available_cols}")

    X = df[available_cols].values
    y = df['species'].values
    y_num = df['species_label'].values

    # ========================================
    # A) STANDARDIZED PCA
    # ========================================
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca_std = PCA()
    X_std_pca = pca_std.fit_transform(X_std)

    print("\n--- Standardized Data PCA ---")
    print("Explained variance ratio:")
    for i, r in enumerate(pca_std.explained_variance_ratio_, 1):
        print(f"  PC{i}: {r:.4f}")

    # Component formulas (loadings)
    print("\nPrincipal Component Formulas (loadings):")
    components_df = pd.DataFrame(
        pca_std.components_,
        columns=available_cols,
        index=[f'PC{i+1}' for i in range(len(pca_std.components_))]
    )
    print(components_df.round(4))

    # ========================================
    # B) NON-STANDARDIZED PCA
    # ========================================
    pca_raw = PCA()
    X_raw_pca = pca_raw.fit_transform(X)

    print("\n--- Non-Standardized Data PCA ---")
    print("Explained variance ratio:")
    for i, r in enumerate(pca_raw.explained_variance_ratio_, 1):
        print(f"  PC{i}: {r:.4f}")

    # ========================================
    # C) VISUALIZATION COMPARISON
    # ========================================
    # 2D comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(x=X_std_pca[:, 0], y=X_std_pca[:, 1], hue=y, palette='Set2', ax=axes[0], s=60)
    axes[0].set_title("Standardized: PC1 vs PC2")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sns.scatterplot(x=X_raw_pca[:, 0], y=X_raw_pca[:, 1], hue=y, palette='Set2', ax=axes[1], s=60)
    axes[1].set_title("Non-Standardized: PC1 vs PC2")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.show()

    # 3D comparison
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    for s in species_unique:
        mask = y == s
        ax1.scatter(X_std_pca[mask, 0], X_std_pca[mask, 1], X_std_pca[mask, 2], label=s, s=40)
    ax1.set_title("Standardized: PC1-PC2-PC3")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    for s in species_unique:
        mask = y == s
        ax2.scatter(X_raw_pca[mask, 0], X_raw_pca[mask, 1], X_raw_pca[mask, 2], label=s, s=40)
    ax2.set_title("Non-Standardized: PC1-PC2-PC3")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("\nObservation:")
    print("  - Standardization usually improves species separation because features are on different scales")
    print("    (e.g., body_mass_g is in grams, others in mm).")
    print("  - Without standardization, high-magnitude features dominate the principal components.")

    # ========================================
    # D) ALTERNATIVE SCALER: RobustScaler
    # ========================================
    print("\n--- Alternative Scaler: RobustScaler ---")
    robust = RobustScaler()
    X_robust = robust.fit_transform(X)
    pca_robust = PCA(n_components=2)
    X_robust_pca = pca_robust.fit_transform(X_robust)

    plot_2d_projection(X_robust_pca, y, title="RobustScaler + PCA 2D", palette='Set2')

    print("Why RobustScaler?")
    print("  - It uses median and IQR instead of mean and std.")
    print("  - Less sensitive to outliers, which is useful if penguin data has extreme values.")
    print("  - Results are often similar to StandardScaler for well-behaved data, but more robust.")


# ================================================================================
# SECTION 4: PCA ON AMP DATA (Exercise 4)
# ================================================================================

def exercise_4_amp():
    """
    Exercise 4: PCA on AMP (Antimicrobial Peptide) data.
    Since no specific file is provided, this function:
      - Generates synthetic AMP-like data (molecular descriptors)
      - Demonstrates the full PCA pipeline
      - Writes observations and conclusions

    In a real scenario, replace the synthetic generation with:
      df = pd.read_csv('your_amp_data.csv')
    """
    print("\n" + "="*60)
    print("EXERCISE 4: PCA ON AMP DATA (Synthetic Demo)")
    print("="*60)

    np.random.seed(42)
    n_samples = 200

    # Simulate AMP features: molecular weight, hydrophobicity, charge, etc.
    # In real data, these would come from peptide descriptor calculations
    mw = np.random.normal(2500, 800, n_samples)          # Molecular weight (Da)
    gravy = np.random.normal(-0.5, 1.2, n_samples)       # Hydrophobicity
    charge = np.random.normal(2, 3, n_samples)           # Net charge
    aromaticity = np.random.beta(2, 5, n_samples)        # Aromatic content
    instability = np.random.normal(40, 15, n_samples)    # Instability index

    # Create synthetic classes: active vs inactive AMPs
    # Active peptides tend to have higher charge and moderate hydrophobicity
    activity_score = (charge * 0.3 + (-gravy) * 0.4 + instability * 0.01)
    labels = np.where(activity_score > np.median(activity_score), 'active', 'inactive')

    df_amp = pd.DataFrame({
        'molecular_weight': mw,
        'hydrophobicity': gravy,
        'net_charge': charge,
        'aromaticity': aromaticity,
        'instability_index': instability,
        'activity': labels
    })

    print("Synthetic AMP dataset (first 5 rows):")
    print(df_amp.head())

    features = ['molecular_weight', 'hydrophobicity', 'net_charge', 'aromaticity', 'instability_index']
    X = df_amp[features].values
    y = df_amp['activity'].values

    # CRITICAL: Standardize because scales differ wildly (MW ~2500, gravy ~-0.5)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    print("\nExplained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")

    plot_2d_projection(X_pca, y, title="PCA on AMP Data (Synthetic)", palette=['blue', 'orange'])

    print("\nObservations & Conclusions:")
    print("  1. Standardization is MANDATORY: Molecular weight (thousands) would dominate")
    print("     over hydrophobicity (-1 to 1) without scaling.")
    print("  2. PCA reveals which physicochemical properties contribute most to variance.")
    print("  3. If active/inactive peptides separate well in PC space, PCA can be used")
    print("     as a preprocessing step for classification.")
    print("  4. For real AMP data, replace synthetic generation with actual descriptor files.")


# ================================================================================
# SECTION 5: KERNEL PCA (Exercises)
# ================================================================================

def exercise_kernel_pca():
    """
    Kernel PCA exercises:
      - Demonstrate failure of linear PCA on make_moons
      - Show Kernel PCA (RBF) success on make_moons
      - Apply Kernel PCA to Iris, Breast Cancer, AMP, and MNIST
      - Experiment with different kernels and gamma values
      - Reconstruction comparison: PCA vs KernelPCA inverse_transform
    """
    print("\n" + "="*60)
    print("KERNEL PCA EXERCISES")
    print("="*60)

    # ----------------------------------------
    # 5.1: Make Moons (Linear PCA fails)
    # ----------------------------------------
    print("\n--- 5.1: Make Moons Dataset ---")
    X_moon, y_moon = make_moons(n_samples=500, noise=0.02, random_state=417)

    # Linear PCA
    pca_moon = PCA(n_components=2)
    X_moon_pca = pca_moon.fit_transform(X_moon)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap='coolwarm', edgecolors='k')
    axes[0].set_title("Original Make Moons")

    axes[1].scatter(X_moon_pca[:, 0], X_moon_pca[:, 1], c=y_moon, cmap='coolwarm', edgecolors='k')
    axes[1].set_title("Linear PCA (Fails)")

    # Kernel PCA with RBF
    kpca_moon = KernelPCA(kernel='rbf', gamma=15, n_components=2, fit_inverse_transform=True)
    X_moon_kpca = kpca_moon.fit_transform(X_moon)

    axes[2].scatter(X_moon_kpca[:, 0], X_moon_kpca[:, 1], c=y_moon, cmap='coolwarm', edgecolors='k')
    axes[2].set_title("Kernel PCA (RBF, gamma=15)")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------
    # 5.2: Kernel PCA on Iris (experiment with kernels)
    # ----------------------------------------
    print("\n--- 5.2: Kernel PCA on Iris (kernel comparison) ---")
    df_iris = sns.load_dataset('iris')
    X_iris = StandardScaler().fit_transform(df_iris[['sepal_length', 'sepal_width',
                                                       'petal_length', 'petal_width']].values)
    y_iris = df_iris['species'].values

    kernels = ['linear', 'rbf', 'poly', 'cosine']
    fig, axes = plt.subplots(1, len(kernels), figsize=(16, 4))

    for ax, kernel in zip(axes, kernels):
        gamma_val = 0.1 if kernel == 'rbf' else None
        kpca = KernelPCA(kernel=kernel, gamma=gamma_val, n_components=2)
        X_k = kpca.fit_transform(X_iris)
        sns.scatterplot(x=X_k[:, 0], y=X_k[:, 1], hue=y_iris, palette='Set1', ax=ax, legend=False)
        ax.set_title(f"Kernel: {kernel}")
    plt.suptitle("KernelPCA on Iris with Different Kernels")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------
    # 5.3: MNIST Dataset (PCA vs KernelPCA)
    # ----------------------------------------
    print("\n--- 5.3: MNIST Dataset ---")
    print("Loading MNIST (this may take a moment)...")

    try:
        # Load subset of MNIST for speed
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X_mnist = mnist.data[:3000]  # Use subset for speed
        y_mnist = mnist.target[:3000].astype(int)

        # Normalize pixel values
        X_mnist = X_mnist / 255.0

        print(f"MNIST subset shape: {X_mnist.shape}")

        # PCA
        pca_mnist = PCA()
        pca_mnist.fit(X_mnist)

        print("\nExplained variance analysis:")
        find_n_components_for_variance(pca_mnist)
        plot_explained_variance(pca_mnist, "MNIST PCA Explained Variance")

        # Reconstruction experiment
        n_components_list = [find_n_components_for_variance(pca_mnist, [0.85, 0.90, 0.95, 0.99])[t]
                             for t in [0.85, 0.90, 0.95, 0.99]]

        # Draw original samples
        sample_idx = [0, 1, 2]
        fig, axes = plt.subplots(len(sample_idx), len(n_components_list)+1, figsize=(12, 6))

        for i, idx in enumerate(sample_idx):
            axes[i, 0].imshow(X_mnist[idx].reshape(28, 28), cmap='gray')
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            for j, n_comp in enumerate(n_components_list):
                pca_temp = PCA(n_components=n_comp)
                X_reduced = pca_temp.fit_transform(X_mnist)
                X_rec = pca_temp.inverse_transform(X_reduced)
                axes[i, j+1].imshow(X_rec[idx].reshape(28, 28), cmap='gray')
                axes[i, j+1].set_title(f"PCA {n_comp}")
                axes[i, j+1].axis('off')

        plt.suptitle("MNIST Reconstruction with PCA (Different Component Counts)")
        plt.tight_layout()
        plt.show()

        # KernelPCA reconstruction (using smaller subset due to computational cost)
        print("\nKernelPCA reconstruction (using 500 samples for speed)...")
        X_mnist_small = X_mnist[:500]
        n_comp_kpca = 50  # Fixed number for comparison

        kpca_mnist = KernelPCA(kernel='rbf', gamma=0.01, n_components=n_comp_kpca,
                               fit_inverse_transform=True)
        X_k_reduced = kpca_mnist.fit_transform(X_mnist_small)
        X_k_rec = kpca_mnist.inverse_transform(X_k_reduced)

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axes[0, i].imshow(X_mnist_small[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            axes[1, i].imshow(X_k_rec[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title(f"KPCA {n_comp_kpca}")
            axes[1, i].axis('off')
        plt.suptitle("MNIST: Original vs KernelPCA Reconstruction")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"MNIST loading failed: {e}")
        print("Skipping MNIST section. Install sklearn with openml support if needed.")


# ================================================================================
# SECTION 6: THE GREAT DIMENSIONALITY REDUCTION SHOWDOWN
# ================================================================================

def exercise_showdown():
    """
    Final comparison: PCA vs t-SNE vs UMAP
    Datasets used: Iris, Breast Cancer (subset), and optionally MNIST subset.
    Reports:
      - Which algorithm gives cleanest separation
      - Computation time differences
      - t-SNE perplexity sensitivity (low vs high)
    """
    print("\n" + "="*60)
    print("THE GREAT DIMENSIONALITY REDUCTION SHOWDOWN")
    print("="*60)

    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with: pip install umap-learn")
        return

    # Dataset 1: Iris
    print("\n--- Showdown 1: Iris Dataset ---")
    df_iris = sns.load_dataset('iris')
    X_iris = StandardScaler().fit_transform(df_iris.iloc[:, :4].values)
    y_iris = df_iris['species'].values
    plot_comparison(X_iris, y_iris, title_suffix="Iris", perplexity=30)

    # Dataset 2: Breast Cancer (subset for speed)
    print("\n--- Showdown 2: Breast Cancer Dataset ---")
    data_cancer = load_breast_cancer()
    X_cancer = StandardScaler().fit_transform(data_cancer.data)
    y_cancer = np.array(['malignant' if t == 0 else 'benign' for t in data_cancer.target])
    plot_comparison(X_cancer, y_cancer, title_suffix="Breast Cancer", perplexity=30)

    # Dataset 3: MNIST subset (if available)
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X_mnist = mnist.data[:2000] / 255.0
        y_mnist = mnist.target[:2000].astype(int)
        print("\n--- Showdown 3: MNIST Subset (2000 samples) ---")
        plot_comparison(X_mnist, y_mnist, title_suffix="MNIST Subset", perplexity=30)
    except Exception as e:
        print(f"MNIST showdown skipped: {e}")

    # t-SNE perplexity sensitivity experiment
    print("\n--- t-SNE Perplexity Sensitivity ---")
    df_iris = sns.load_dataset('iris')
    X_iris = StandardScaler().fit_transform(df_iris.iloc[:, :4].values)
    y_iris_num = df_iris['species'].astype('category').cat.codes.values

    perplexities = [2, 5, 30, 100]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(16, 4))

    for ax, perp in zip(axes, perplexities):
        print(f"Running t-SNE with perplexity={perp}...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X_iris)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_iris_num, cmap='Set1', s=50, alpha=0.7)
        ax.set_title(f"t-SNE perplexity={perp}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Effect of Perplexity on t-SNE (Iris Dataset)")
    plt.tight_layout()
    plt.show()

    print("\n--- REPORT SUMMARY ---")
    print("1. Cleanest separation:")
    print("   - UMAP and t-SNE usually provide better class separation than linear PCA")
    print("     for non-linearly separable data.")
    print("   - PCA is fastest but only captures linear variance.")
    print("2. Computation time:")
    print("   - PCA: Fastest (analytical solution, no iterations).")
    print("   - UMAP: Fast, scalable, good for large datasets.")
    print("   - t-SNE: Slowest, especially with large perplexity or dataset size.")
    print("3. Perplexity sensitivity:")
    print("   - Low perplexity (e.g., 2): Focuses on very local neighborhoods,")
    print("     can create many small, tight clusters (sometimes artificial).")
    print("   - High perplexity (e.g., 100): Considers global structure more,")
    print("     but may compress distinct clusters together.")
    print("   - Perplexity=30 is a common default for Iris-sized datasets.")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    """
    Run all exercises sequentially.
    Comment out any function call to skip a specific exercise.
    """
    print("DIMENSIONALITY REDUCTION LAB - STARTING")


    # Run exercises
    exercise_1_iris()
    exercise_2_breast_cancer()
    exercise_3_penguins()
    exercise_4_amp()
    exercise_kernel_pca()
    exercise_showdown()

    print("\n" + "="*60)
    print("ALL EXERCISES COMPLETED")
    print("="*60)