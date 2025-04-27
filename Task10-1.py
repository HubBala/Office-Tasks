# Task10 - K-means Clustering - Enhanced Visualization and Interpretation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For pair plots

# Load the dataset
df = pd.read_csv('Lung Cancer Dataset Innv.csv')
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})
df = df.dropna()

# Separate features (X) and target (if any, though K-Means is unsupervised)
X = df.copy()
if 'Cluster' in X.columns:
    X = X.drop('Cluster', axis=1, errors='ignore') # Remove previous cluster assignments

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Create a DataFrame for easier manipulation

# Elbow method for finding optimal k
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
'''
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show() '''

# Apply K-means Clustering with the chosen optimal k
optimal_k = 3  # You can adjust this based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
X_scaled_df['Cluster'] = df['Cluster'] # Add cluster labels to the scaled DataFrame

print("\nCluster Assignments:\n", df['Cluster'].value_counts().sort_index())

# Analyze Cluster Characteristics
print("\nMean Characteristics of Each Cluster (Scaled Data):")
print(X_scaled_df.groupby('Cluster').mean())

print("\nMean Characteristics of Each Cluster (Original Scale - Interpretation):")
original_means = df.groupby('Cluster').mean()
print(original_means)

# Interpretation of Cluster Means (Adapt based on your features)
print("\nInterpretation of Clusters:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    cluster_data = original_means.loc[cluster]
    # Example interpretations - ADJUST THESE BASED ON YOUR ACTUAL FEATURES
    print(f"  Average Age: {cluster_data['AGE']:.2f}")
    print(f"  Prevalence of Pulmonary Disease: {cluster_data['PULMONARY_DISEASE']:.2f}")
    # Add interpretations for other relevant features in your dataset
    # Consider symptoms, risk factors, etc.

# Visualization using Pair Plots (for a multi-dimensional view)
plt.figure(figsize=(12, 12))
sns.pairplot(df, hue='Cluster', diag_kind='kde')
plt.suptitle("Pair Plot of Clusters", y=1.02)
plt.show()

# Visualization of Cluster Centers in the Scaled Space (First Two Components)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
centers_scaled = kmeans.cluster_centers_
plt.scatter(centers_scaled[:, 0], centers_scaled[:, 1], marker='^', c='red', s=200, edgecolors='black', label='Cluster Centers')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.legend(loc='upper right')
plt.title("K-Means Clustering (Scaled Data - First Two Components)")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.show()