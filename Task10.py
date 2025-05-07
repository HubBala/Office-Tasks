# Task10 - K-means Clustering

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('Lung Cancer Dataset Innv.csv')
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

df = df.dropna()

# Standardization 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Elbow method for the K
inertia=[]
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state= 42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
'''
# plot Elbow
plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k) ")
plt.ylabel("inertia")
plt.title("Elbow Method")
plt.show() '''


# k-means Clustering
optimal_k=3 
start_time = time.time()

kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = 10 )
df['Cluster'] = kmeans.fit_predict(df_scaled)

# print(df.head())
end_time = time.time()
print(f"Clustering completed in: {end_time - start_time:.6f} seconds")

for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Characteristics:\n", df[df['Cluster'] == cluster].mean(numeric_only=True))

# plot for cluster
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=df['Cluster'], cmap='viridis')

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='^', c='red', s=200, edgecolors='black', label='Centers')

plt.legend(*scatter.legend_elements(), title="Clusters") # Add legend for clusters
plt.legend(loc='upper right') # Ensure cluster centers legend is visible
plt.title("K-Means Clustering on Scaled Data")
plt.xlabel("Scaled Feature 1") # More accurate label
plt.ylabel("Scaled Feature 2") # More accurate label
plt.show()
