# segmentation.py

import sys

# --- Ensure required libraries are installed ---
try:
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
except ModuleNotFoundError as e:
    print(f"❌ Missing library: {e.name}")
    print("👉 Run: pip install pandas scikit-learn matplotlib seaborn")
    sys.exit(1)

# --- Load the clustering data (features only) ---
X = pd.read_csv('segmentation_data.csv').values

# --- Run K-Means clustering ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# --- Load the full dataset with original columns ---
df_full = pd.read_csv('full_data.csv')

# --- Attach cluster labels ---
df_full['Cluster'] = clusters

# --- Save enriched dataset ---
df_full.to_csv('data_with_clusters.csv', index=False)

# --- Compute cluster profiles ---
cluster_profiles = df_full.groupby('Cluster')[['Age', 'Annual_Income_USD', 'Spending_Score']].mean()
cluster_profiles.to_csv('cluster_profiles.csv')

print("✅ Segments created!")
print(cluster_profiles)

# --- Optional: Visualize clusters with PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="Set2", s=60)
plt.title("Customer Segmentation (PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
