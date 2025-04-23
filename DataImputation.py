import pandas as pd 
import numpy as np 

# Creating the dataset
data = { 
    "A" : [1, 2, np.nan, 4], 
    "B" : [5, np.nan, np.nan, 8], 
    "C" : ["cat", "dog", "cat", None] 
} 

df = pd.DataFrame(data) 

# Original data
print("Original Data:")
print(df) 

# Mean Imputation for column A 
df['A_mean_imputed'] = df['A'].fillna(df['A'].mean()) 

# Median Imputation for column B 
df['B_median_imputed'] = df['B'].fillna(df['B'].median()) 

# Mode Imputation for column C 
df['C_mode_imputed'] = df['C'].fillna(df['C'].mode()[0]) 

# Displaying the data after imputation
print("\nData after Imputation:") 
print(df)
