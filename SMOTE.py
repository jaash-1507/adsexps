#pip install numpy pandas matplotlib scikit-learn imbalanced-learn

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_breast_cancer 
from sklearn.decomposition import PCA 
from collections import Counter 

# Load dataset
cancer = load_breast_cancer() 
X = cancer.data 
y = cancer.target 
X = pd.DataFrame(X, columns=cancer.feature_names) 

# Standardize features
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Split dataset with stratified target distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
) 

# Show class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y_train)) 

# Define PCA plot function
def plot_data(X, y, title): 
    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X) 
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5, cmap='coolwarm') 
    plt.title(title) 
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show() 

# Plot before SMOTE
plot_data(X_train, y_train, "Before SMOTE") 

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42) 
X_resampled, y_resampled = smote.fit_resample(X_train, y_train) 

# Show class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_resampled)) 

# Plot after SMOTE
plot_data(X_resampled, y_resampled, "After SMOTE") 

# Display a few samples from the resampled data
print("Sample of resampled dataset:")
print(pd.DataFrame(X_resampled, columns=cancer.feature_names).head())
