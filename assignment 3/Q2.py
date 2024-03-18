import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/weichen/Desktop/CALIFORNIA-housing.csv')

# Selecting longitude and latitude columns
X = data[['longitude', 'latitude']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN clustering
eps = 0.2
min_samples = 50
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
data['cluster'] = dbscan.fit_predict(X_scaled)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='longitude', y='latitude', hue='cluster', palette='viridis')
plt.title('DBSCAN Clustering of California Housing Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

# Boxplot comparison across clusters
numerical_columns = ['median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

plt.figure(figsize=(14, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=data, x='cluster', y=column, palette='viridis')
    plt.title(f'{column} across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(column)
plt.tight_layout()
plt.show()

