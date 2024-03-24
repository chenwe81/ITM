import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/Users/weichen/Desktop/CALIFORNIA-housing.csv')

# Selecting longitude and latitude columns
coordinates = data[['longitude', 'latitude']]

# # Standardize the data
# scaler = StandardScaler()
# coordinates_scaled = scaler.fit_transform(coordinates)

# # Set a random seed for reproducibility
# np.random.seed(42)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=50)
data['cluster_label'] = dbscan.fit_predict(coordinates)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], c=data['cluster_label'], cmap='viridis', s=20)
plt.title('DBSCAN Clustering of California Housing Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Define numerical variables
numerical_vars = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

# Plot boxplots for each variable across clusters
plt.figure(figsize=(15, 10))
# Create subplots
num_rows = 3
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

for i, var in enumerate(numerical_vars):
    row = i // num_cols
    col = i % num_cols
    data.boxplot(column=var, by='cluster_label', grid=False, ax=axes[row, col])
    axes[row, col].set_title(f'{var} across clusters')
    axes[row, col].set_xlabel('Cluster')
    axes[row, col].set_ylabel(var)

# Adjust layout
plt.tight_layout()