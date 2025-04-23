#pip install numpy pandas matplotlib seaborn scikit-learn

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing, load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.cluster import KMeans 
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.preprocessing import StandardScaler 

# 1. Load the California Housing dataset (for regression) 
california = fetch_california_housing() 
X_regression = california.data 
y_regression = california.target 

# 2. Load the Iris dataset (for classification) 
iris = load_iris() 
X_classification = iris.data 
y_classification = iris.target 

# 3. Split the regression dataset into train and test sets 
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.3, random_state=42)

# 4. Split the classification dataset into train and test sets 
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.3, random_state=42)

# 5. Supervised Learning: Linear Regression for California Housing Dataset 
regressor = LinearRegression() 
regressor.fit(X_train_regression, y_train_regression) 

# 6. Predictions for regression model 
y_pred_regression = regressor.predict(X_test_regression) 

# 7. Calculate MSE and MAE for regression 
mse = mean_squared_error(y_test_regression, y_pred_regression) 
mae = mean_absolute_error(y_test_regression, y_pred_regression) 
print(f"Regression Model (California Housing) - MSE: {mse:.2f}, MAE: {mae:.2f}") 

# 8. Unsupervised Learning: KMeans Clustering for Iris Dataset 
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for Iris dataset 
kmeans.fit(X_train_classification) 

# 9. Predictions for clustering 
y_pred_classification = kmeans.predict(X_test_classification) 

# 10. Confusion Matrix for Clustering (Iris dataset) 
cm = confusion_matrix(y_test_classification, y_pred_classification)

# 11. Plot Confusion Matrix for the clustering 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names) 
disp.plot(cmap='Blues') 
plt.title('Confusion Matrix for KMeans Clustering (Iris)') 
plt.show() 

# 12. Visualizing predicted vs actual values for the regression model 
plt.scatter(y_test_regression, y_pred_regression) 
plt.plot([y_test_regression.min(), y_test_regression.max()], 
         [y_test_regression.min(), y_test_regression.max()], 'r--') 
plt.xlabel('Actual Values') 
plt.ylabel('Predicted Values') 
plt.title('Regression: Actual vs Predicted (California Housing)') 
plt.show() 

# 13. Visualizing the clusters for the unsupervised model (Iris) 
plt.scatter(X_test_classification[:, 0], X_test_classification[:, 1], 
            c=y_pred_classification, cmap='viridis') 
plt.title('KMeans Clustering (Unsupervised - Iris)') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.show()
