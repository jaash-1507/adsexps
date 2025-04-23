#pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Create a sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'Age': np.random.randint(18, 60, 100),
    'Salary': np.random.randint(30000, 100000, 100),
    'Experience': np.random.randint(1, 40, 100),
    'Department': np.random.choice(['HR', 'IT', 'Finance', 'Sales'], 100)
})

# Univariate Analysis (Histogram)
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Univariate Analysis - Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Line Chart
df_sorted = df.sort_values(by='Age')
plt.figure(figsize=(6, 4))
plt.plot(df_sorted['Age'], df_sorted['Salary'], color='green')
plt.title('Line Chart - Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Bar Chart (Categorical variable)
plt.figure(figsize=(6, 4))
sns.countplot(x='Department', data=df)
plt.title('Bar Chart - Department Count')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Experience', y='Salary', data=df)
plt.title('Scatter Plot - Experience vs Salary')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.show()

# Multivariate Analysis - Pair Plot (Scatter Matrix)
sns.pairplot(df[['Age', 'Salary', 'Experience']])
plt.suptitle('Scatter Matrix (Pair Plot)', y=1.02)
plt.show()

# Bubble Chart (Scatter plot with size based on Age)
plt.figure(figsize=(6, 4))
plt.scatter(df['Experience'], df['Salary'], s=df['Age'], alpha=0.5, c='red')
plt.title('Bubble Chart - Experience vs Salary (Bubble size = Age)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Density Chart
plt.figure(figsize=(6, 4))
sns.kdeplot(df['Salary'], shade=True, color='purple')
plt.title('Density Plot - Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.show()
