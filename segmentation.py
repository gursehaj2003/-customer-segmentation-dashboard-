import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the clustering data (features only)
X = pd.read_csv('segmentation_data.csv').values

# Run K‑Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Load the full dataset (with original features/columns)
df_full = pd.read_csv('full_data.csv')

# Attach the cluster labels to the full dataset
df_full['Cluster'] = clusters

# Save the enriched dataset with clusters
df_full.to_csv('data_with_clusters.csv', index=False)

# Compute cluster profiles (mean values per cluster)
cluster_profiles = df_full.groupby('Cluster')[['Age', 'Annual_Income_USD', 'Spending_Score']].mean()

# Save cluster profiles
cluster_profiles.to_csv('cluster_profiles.csv')

print("✅ Segments created!")
print(cluster_profiles)
