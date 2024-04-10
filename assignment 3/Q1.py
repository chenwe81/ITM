import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/Users/weichen/Desktop/CALIFORNIA-housing.csv")

X = data[['longitude', 'latitude']].values

kmeans = KMeans(n_clusters=12, random_state=42)
data['cluster_label'] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
for cluster in range(12):
    cluster_data = data[data['cluster_label'] == cluster]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'], label=f'Cluster {cluster}')
plt.title('K-means Clustering of California Housing Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

numerical_columns = ['median_house_value', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
plt.figure(figsize=(14, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    data.boxplot(column, by='cluster_label', ax=plt.gca())
    plt.title(column)
    plt.xlabel('Cluster')
    plt.ylabel(column)
plt.suptitle('')
plt.tight_layout()
plt.show()

