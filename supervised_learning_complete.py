"""
================================================================================
SUPERVISED LEARNING LAB - COMPLETE EXERCISES
Decision Trees & Random Forests
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# =============================================================================
# EXERCISE 1
# =============================================================================

def exercise_1():
    print("=" * 70)
    print("EXERCISE 1: Return of the Penguins")
    print("=" * 70)

    # Load dataset
    penguins = sns.load_dataset('penguins')

    print(f"Original shape: {penguins.shape}")

    # Remove missing values
    penguins = penguins.dropna()

    print(f"After dropna: {penguins.shape}")

    # One-hot encoding
    penguins_encoded = pd.get_dummies(
        penguins,
        columns=['island', 'sex'],
        drop_first=False
    )

    print(f"Encoded columns: {list(penguins_encoded.columns)}")

    # Features / target
    X = penguins_encoded.drop('species', axis=1)
    y = penguins_encoded['species']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training set:\n{y_train.value_counts()}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n>>> RESULTS <<<")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=rf.classes_)

    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(
        cm,
        index=rf.classes_,
        columns=[f"pred_{c}" for c in rf.classes_]
    )

    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    # Feature importance
    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 5 Features:")
    print(importances.head())

    # Plot
    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=importances.values,
        y=importances.index,
        palette="viridis"
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.savefig("ex1_feature_importance.png")
    plt.show()

    # Hyperparameter sweep
    print("\n--- Hyperparameter Sweep ---")

    print(
        f"{'n_est':>6} | "
        f"{'depth':>5} | "
        f"{'min_split':>10} | "
        f"{'train_acc':>10} | "
        f"{'test_acc':>10}"
    )

    print("-" * 65)

    results = []

    for n_est in [10, 50, 100]:
        for depth in [3, 5, 10, None]:
            for min_split in [2, 5, 10]:

                rf_exp = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=min_split,
                    random_state=42
                )

                t0 = time.time()

                rf_exp.fit(X_train, y_train)

                t1 = time.time()

                tr_acc = accuracy_score(
                    y_train,
                    rf_exp.predict(X_train)
                )

                te_acc = accuracy_score(
                    y_test,
                    rf_exp.predict(X_test)
                )

                d_str = str(depth)

                print(
                    f"{n_est:6d} | "
                    f"{d_str:>5} | "
                    f"{min_split:10d} | "
                    f"{tr_acc:10.3f} | "
                    f"{te_acc:10.3f}"
                )

                results.append((tr_acc, te_acc))

    print("\nExercise 1 complete.\n")


# =============================================================================
# EXERCISE 2
# =============================================================================

def exercise_2():
    print("=" * 70)
    print("EXERCISE 2: Hyperparameter Tuning")
    print("=" * 70)

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    depths = list(range(1, 16))

    train_accs = []
    test_accs = []

    for depth in depths:

        dt = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        dt.fit(X_train, y_train)

        train_accs.append(
            accuracy_score(
                y_train,
                dt.predict(X_train)
            )
        )

        test_accs.append(
            accuracy_score(
                y_test,
                dt.predict(X_test)
            )
        )

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(
        depths,
        train_accs,
        'o-',
        label='Training Accuracy'
    )

    plt.plot(
        depths,
        test_accs,
        's-',
        label='Test Accuracy'
    )

    best_depth = depths[np.argmax(test_accs)]

    plt.axvline(
        x=best_depth,
        linestyle='--',
        label=f'Best depth = {best_depth}'
    )

    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")

    plt.title(
        "Decision Tree Accuracy vs max_depth\n"
        "(Breast Cancer Dataset)"
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig("ex2_tree_depth_tuning.png")
    plt.show()

    print("\n>>> RESULTS <<<")

    for d, tr, te in zip(depths, train_accs, test_accs):
        print(
            f"Depth={d:2d} | "
            f"Train={tr:.4f} | "
            f"Test={te:.4f}"
        )

    print(f"\nBest depth: {best_depth}")

    print("\nExercise 2 complete.\n")


# =============================================================================
# EXERCISE 3
# =============================================================================

def exercise_3():
    print("=" * 70)
    print("EXERCISE 3: Random Forest vs PCA")
    print("=" * 70)

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Original RF
    rf_original = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_original.fit(X_train, y_train)

    y_pred_orig = rf_original.predict(X_test)

    acc_original = accuracy_score(
        y_test,
        y_pred_orig
    )

    print(f"\nOriginal RF Accuracy: {acc_original:.4f}")

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=2)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("\nPCA Explained Variance:")
    print(pca.explained_variance_ratio_)

    # RF on PCA
    rf_pca = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_pca.fit(X_train_pca, y_train)

    y_pred_pca = rf_pca.predict(X_test_pca)

    acc_pca = accuracy_score(
        y_test,
        y_pred_pca
    )

    print(f"\nRF on PCA Accuracy: {acc_pca:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))

    colors = np.where(
        y_train == 0,
        'red',
        'blue'
    )

    plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=colors,
        alpha=0.6
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title("PCA Projection")

    plt.tight_layout()
    plt.savefig("ex3_pca_projection.png")
    plt.show()

    print("\nExercise 3 complete.\n")


# =============================================================================
# EXERCISE 4
# =============================================================================

def exercise_4():
    print("=" * 70)
    print("EXERCISE 4: AMP Classifier")
    print("=" * 70)

    np.random.seed(42)

    n_samples = 600

    # Synthetic dataset
    charge = np.random.normal(1.5, 2.5, n_samples)

    hydrophobicity = np.random.normal(
        0.35,
        0.25,
        n_samples
    )

    molecular_weight = np.random.normal(
        4500,
        1800,
        n_samples
    )

    isoelectric_point = np.random.normal(
        8.5,
        2.0,
        n_samples
    )

    aliphatic_index = np.random.normal(
        55,
        22,
        n_samples
    )

    boman_index = np.random.normal(
        1.1,
        0.6,
        n_samples
    )

    instability_index = np.random.normal(
        32,
        14,
        n_samples
    )

    aromaticity = np.random.beta(
        2.5,
        6,
        n_samples
    )

    gravy = np.random.normal(
        -0.3,
        0.9,
        n_samples
    )

    score = (
        charge * 0.25
        + hydrophobicity * 1.8
        + (isoelectric_point - 7) * 0.4
        - instability_index * 0.015
    )

    y = (
        score > np.percentile(score, 55)
    ).astype(int)

    amp_df = pd.DataFrame({
        'charge': charge,
        'hydrophobicity': hydrophobicity,
        'molecular_weight': molecular_weight,
        'isoelectric_point': isoelectric_point,
        'aliphatic_index': aliphatic_index,
        'boman_index': boman_index,
        'instability_index': instability_index,
        'aromaticity': aromaticity,
        'gravy': gravy,
        'is_amp': y
    })

    print(f"Dataset shape: {amp_df.shape}")
    print(f"Class distribution:\n{amp_df['is_amp'].value_counts()}")

    X = amp_df.drop('is_amp', axis=1)
    y = amp_df['is_amp']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Full model
    rf_full = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )

    rf_full.fit(X_train, y_train)

    y_pred_full = rf_full.predict(X_test)

    acc_full = accuracy_score(
        y_test,
        y_pred_full
    )

    print(f"\nFull Feature Accuracy: {acc_full:.4f}")

    # PCA
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=2)

    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    rf_pca = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )

    rf_pca.fit(X_train_pca, y_train)

    y_pred_pca = rf_pca.predict(X_test_pca)

    acc_pca = accuracy_score(
        y_test,
        y_pred_pca
    )

    print(f"PCA Accuracy: {acc_pca:.4f}")

    # Feature importance
    importances = pd.Series(
        rf_full.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFeature Importances:")
    print(importances)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=importances.values,
        y=importances.index,
        palette="magma"
    )

    plt.title("AMP Feature Importance")

    plt.tight_layout()
    plt.savefig("ex4_amp_feature_importance.png")
    plt.show()

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        rf_full,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    print("\nCross Validation Scores:")
    print(scores)

    print(f"Mean CV Accuracy: {scores.mean():.4f}")

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_full))

    print("\nExercise 4 complete.\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

    print("=" * 70)
    print("ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("=" * 70)