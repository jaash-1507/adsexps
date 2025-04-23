import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the tips dataset
tips = sns.load_dataset('tips')

# Extract the 'total_bill' column
data = tips['total_bill']

# Calculate quartiles
Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)
Q3 = np.percentile(data, 75)

# Print the quartiles
print(f"Q1: {Q1}, Q2: {Q2}, Q3: {Q3}")

# Calculate Bowley's coefficient of skewness
B = (Q1 + Q3 - 2 * Q2) / (Q3 - Q1)

# Plot the boxplot
sns.boxplot(x=data)
plt.title('Boxplot of Total Bill')
plt.annotate(f'Bowley Skewness: {B:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)
plt.show()
