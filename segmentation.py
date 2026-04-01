import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the clustering data (features only in segmentation_data.csv)
X = pd.read_csv('segmentation_data.csv').values

# Run K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Load the full dataset with original columns
df_full = pd.read_csv('full_data.csv')

# Attach cluster labels
df_full['Cluster'] = clusters

# Save enriched dataset
df_full.to_csv('data_with_clusters.csv', index=False)

# Compute cluster profiles
cluster_profiles = df_full.groupby('Cluster')[['Age', 'Annual_Income_USD', 'Spending_Score']].mean()
cluster_profiles.to_csv('cluster_profiles.csv')

print("✅ Segments created!")
print(cluster_profiles)
