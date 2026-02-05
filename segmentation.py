import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

X = pd.read_csv('segmentation_data.csv').values
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df_full = pd.read_csv('full_data.csv')
df_full['Cluster'] = clusters
df_full.to_csv('data_with_clusters.csv', index=False)

cluster_profiles = df_full.groupby('Cluster')[['Age', 'Annual_Income_USD', 'Spending_Score']].mean()
cluster_profiles.to_csv('cluster_profiles.csv')
print("✅ Segments created!")
print(cluster_profiles)
