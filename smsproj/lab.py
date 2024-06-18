# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create a DataFrame for the provided dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female'],
    'Age': [19, 21, 20, 23, 31],
    'AnnualIncome': [15, 15, 16, 16, 31],
    'SpendingScore': [39, 81, 6, 40, 40]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Original Dataset:")
print(df)

# Preprocess the data
# Convert Gender to numerical values (0 for Female, 1 for Male)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Select relevant features for clustering
X = df[['Age', 'AnnualIncome', 'SpendingScore']].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (K)
# Use the Elbow method to find the optimal K
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o', linestyle='--')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show()

# From the elbow curve, select the optimal K
# Let's say K=3 in this case

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(df.loc[df['Cluster'] == 0, 'PC1'], df.loc[df['Cluster'] == 0, 'PC2'], label='Cluster 0')
plt.scatter(df.loc[df['Cluster'] == 1, 'PC1'], df.loc[df['Cluster'] == 1, 'PC2'], label='Cluster 1')
plt.scatter(df.loc[df['Cluster'] == 2, 'PC1'], df.loc[df['Cluster'] == 2, 'PC2'], label='Cluster 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segmentation')
plt.legend()
plt.show()
