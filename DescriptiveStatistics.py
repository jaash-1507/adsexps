import pandas as pd 
import seaborn as sns 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 

# Load the dataset
iris = pd.read_csv('iris.csv') 
print(iris.head()) 

# Select numeric columns
numeric_columns = iris.select_dtypes(include=[np.number]).columns 

# Compute summary statistics
summary_stats = {} 
for col in numeric_columns: 
    summary_stats[col] = { 
        'mean': iris[col].mean(), 
        'mode': iris[col].mode()[0], 
        'median': iris[col].median(), 
        'skewness': iris[col].skew() 
    } 

# Print summary statistics
for col, stats in summary_stats.items(): 
    print(f"Summary statistics for {col}:") 
    for stat, value in stats.items(): 
        print(f"  {stat}: {value}") 
    print() 

# KDE Plots
plt.figure(figsize=(12, 8)) 
for idx, col in enumerate(numeric_columns, 1): 
    plt.subplot(2, 2, idx) 
    sns.kdeplot(iris[col], shade=True) 
    plt.axvline(summary_stats[col]['mean'], color='r', linestyle='--', label=f"Mean: {summary_stats[col]['mean']:.2f}") 
    plt.axvline(summary_stats[col]['median'], color='g', linestyle='-', label=f"Median: {summary_stats[col]['median']:.2f}") 
    plt.axvline(summary_stats[col]['mode'], color='b', linestyle=':', label=f"Mode: {summary_stats[col]['mode']:.2f}") 
    plt.title(f'KDE plot for {col}') 
    plt.legend() 

plt.tight_layout() 
plt.show()
