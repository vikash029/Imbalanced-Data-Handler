import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_samples=1000, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Plot original class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=y_train)
plt.title("Original Class Distribution")
plt.show()

# Handling Imbalance with SMOTE (Oversampling)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Handling Imbalance with Random Undersampling
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Handling Imbalance with Hybrid (SMOTE + Tomek Links)
smote_tomek = SMOTETomek(random_state=42)
X_train_hybrid, y_train_hybrid = smote_tomek.fit_resample(X_train, y_train)

# Train a model with class weighting
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Plot Resampled Class Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=y_train_smote)
plt.title("Resampled Class Distribution (SMOTE)")
plt.show()
