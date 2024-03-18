import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/weichen/Desktop/CALIFORNIA-housing.csv')

# Selecting longitude and latitude columns
X_geo = data[['longitude', 'latitude']]

# Selecting columns for k-means clustering
X = data[['median_income', 'median_house_value']]

# Run K-means clustering
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='longitude', y='latitude', hue='cluster', palette='viridis')
plt.title('K-Means Clustering of California Housing Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

# Boxplot comparison across clusters
numerical_columns = ['median_income', 'median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households']

plt.figure(figsize=(14, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=data, x='cluster', y=column, palette='viridis')
    plt.title(f'{column} across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(column)
plt.tight_layout()
plt.show()