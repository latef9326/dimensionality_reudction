"""
XAI Analysis: Random Forest Classification on Synthetic Gene Expression Data
Techniques: SHAP + Permutation Feature Importance
Author: Lateef Hanus , Michal Chojnacki
Date: 2026-06-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. DATA GENERATION
# ------------------------------------------------------------------
X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=8,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=[0.6, 0.4],
    flip_y=0.05,
    random_state=42
)

feature_names = [f"GEN_{i+1:02d}" for i in range(30)]
df = pd.DataFrame(X, columns=feature_names)
df['Disease'] = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 2. MODEL TRAINING
# ------------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))

# ------------------------------------------------------------------
# 3. XAI TECHNIQUE 1: SHAP
# ------------------------------------------------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# For binary classification, select class 1 (Disease)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, 
                  max_display=15, show=False)
plt.title('SHAP Summary Plot (Beeswarm)')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

# SHAP Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, 
                  plot_type="bar", max_display=15, show=False)
plt.title('SHAP Feature Importance (Mean |SHAP value|)')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# SHAP Waterfall for single prediction
expected_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
plt.figure(figsize=(10, 6))
shap.plots._waterfall.waterfall_legacy(
    expected_value=expected_val,
    shap_values=shap_values_class1[0],
    features=X_test[0],
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title('SHAP Waterfall Plot — Single Prediction')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 4. XAI TECHNIQUE 2: PERMUTATION FEATURE IMPORTANCE
# ------------------------------------------------------------------
perm_importance = permutation_importance(
    rf_model, X_test, y_test, 
    n_repeats=50, 
    random_state=42, 
    scoring='accuracy',
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=True)

plt.figure(figsize=(10, 8))
top_perm = perm_df.tail(15)
plt.barh(top_perm['feature'], top_perm['importance_mean'], 
         xerr=top_perm['importance_std'], capsize=3, color='coral', edgecolor='black')
plt.xlabel('Decrease in Accuracy')
plt.title('Permutation Feature Importance (Top 15)')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 5. COMPARISON
# ------------------------------------------------------------------
shap_imp = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': np.abs(shap_values_class1).mean(axis=0)
})
perm_imp = pd.DataFrame({
    'feature': feature_names,
    'perm_importance': perm_importance.importances_mean
})
comparison = shap_imp.merge(perm_imp, on='feature').sort_values('shap_importance', ascending=False)
print("\nTop 10 features comparison:")
print(comparison.head(10))
