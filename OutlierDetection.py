#pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# Generate synthetic data with outliers
np.random.seed(42)
data = {'values': np.append(np.random.normal(50, 15, 100), [200, 220, 250, 270])}
df = pd.DataFrame(data)

# Z-score method
def remove_outliers_zscore(df, column, threshold=2.5):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

# IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Distance-based: k-NN distance
def detect_outliers_knn(df, column, k=5):
    values = df[[column]]
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(values)
    distances, _ = nbrs.kneighbors(values)
    dist_scores = distances[:, -1]
    threshold = np.percentile(dist_scores, 95)
    return df[dist_scores < threshold]

# Density-based: Local Outlier Factor (LOF)
def detect_outliers_lof(df, column, n_neighbors=20):
    values = df[[column]]
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    y_pred = lof.fit_predict(values)
    return df[y_pred == 1]  # 1 = inlier, -1 = outlier

# Cleaned datasets
df_zscore = remove_outliers_zscore(df, 'values')
df_iqr = remove_outliers_iqr(df, 'values')
df_knn = detect_outliers_knn(df, 'values', k=5)
df_lof = detect_outliers_lof(df, 'values', n_neighbors=20)

# Plotting
plt.figure(figsize=(12, 6))
plt.hist(df['values'], bins=30, alpha=0.4, label='Original', color='blue')
plt.hist(df_zscore['values'], bins=30, alpha=0.5, label='Z-Score', color='green')
plt.hist(df_iqr['values'], bins=30, alpha=0.5, label='IQR', color='red')
plt.hist(df_knn['values'], bins=30, alpha=0.5, label='k-NN Distance', color='orange')
plt.hist(df_lof['values'], bins=30, alpha=0.5, label='LOF Density', color='purple')
plt.legend()
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Outlier Detection Comparison (Z-Score, IQR, k-NN, LOF)')
plt.grid(True)
plt.show()
