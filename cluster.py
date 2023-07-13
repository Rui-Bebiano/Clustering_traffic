import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#-----------------------------------------------------------------------------------------------------------------------
#   PART I: PCA
#-----------------------------------------------------------------------------------------------------------------------


# Task 1: Import the dataset
df = pd.read_csv('Data/Sublancos.csv')
#print(df.head())
# Saving IDs and creating a color dictionary
colcol='ID'
condition = ~df[colcol].isin(['A12_01'])
df.loc[condition, colcol] = 'Outra'
#print(df.head())
unique_ids = df[colcol].unique()
colors = {id: plt.cm.tab10(i/len(unique_ids)) for i, id in enumerate(unique_ids)}

# Remove the ID column for PCA analysis
df_numeric = df.drop(['ID','VIA','CONC','PORT'], axis=1)
#print(df.head())

# Task 3: Perform PCA analysis
pca = PCA()
df_pca = pca.fit_transform(df_numeric)

# (i) The eigenvalues
explained_variance_ratio = pca.explained_variance_ratio_

# (ii) The scree plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.grid(True)
plt.show()

# (iii) The loading scores
loadings = pca.components_
df_loadings = pd.DataFrame(loadings, columns=df_numeric.columns)

# (iv) The score plot for PC1 and PC2
plt.figure(figsize=(8,6))
for id in unique_ids:
    plt.scatter(df_pca[df[colcol] == id, 0], df_pca[df[colcol] == id, 1], color=colors[id], label=id, s=15, alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()
"""
# (v) The biplot
def biplot(score, coeff, df, colors, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    for id in df[colcol].unique():
        plt.scatter(xs[df[colcol] == id] * scalex, ys[df[colcol] == id] * scaley, color=colors[id], label=id, s=12, alpha=0.4)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color='green', ha='center', va='center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('PC{}'.format(1))
    plt.ylabel('PC{}'.format(2))
    plt.grid()

plt.figure(figsize=(8,6))
biplot(df_pca[:,0:2], np.transpose(pca.components_[0:2, :]), df, colors, labels=df_numeric.columns)
plt.legend()
plt.show()
"""
# (vi) A heatmap of the correlations between variables and PCs
sns.heatmap(pd.DataFrame(pca.components_,columns=df_numeric.columns).transpose(), annot=True)
plt.show()

# Task 4: Save (i) and (iii) into a csv file
eigenvalues = pd.DataFrame(explained_variance_ratio, columns=['Eigenvalue'])
eigenvalues.to_csv('Outputs/eigen.csv', index=False)
df_loadings.to_csv('Outputs/loadings.csv', index=False)


#-----------------------------------------------------------------------------------------------------------------------
#   PART II: CLUSTERING
#-----------------------------------------------------------------------------------------------------------------------

# 1) For k=1 to 10, use k-means to do clustering analyses of the objects on the PC1-PC2 plane
sse = []
silhouette_scores = []
K = range(1,11)

for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(df_pca[:, :2])
    sse.append(kmeans.inertia_)
    if k > 1:  # silhouette_score requires more than one cluster
        silhouette_scores.append(silhouette_score(df_pca[:, :2], kmeans.labels_))

# 2) Plot the SSE and the Silhouette score as a function of k
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# SSE plot
ax1.plot(K, sse, 'bx-')
ax1.set_xlabel('k')
ax1.set_ylabel('SSE')
ax1.set_title('SSE vs. k')

# Silhouette Score plot
ax2.plot(K[1:], silhouette_scores, 'bx-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs. k')

plt.show()

# 3) For k=N_clusters, plot the PC1-PC2 plane with the objects in the respective clusters
N_clusters = 4
kmeans = KMeans(n_clusters=N_clusters, n_init=10, random_state=0).fit(df_pca[:, :2])
df_pca[0,:]=-df_pca[0,:]
plt.figure(figsize=(10, 7))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=300, c='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Objects in Respective Clusters (k={})'.format(N_clusters))
plt.grid()
plt.show()

# 4) Save into csv file
# 4.1) 'ID' and 'Cluster'
df_cluster = pd.DataFrame({'ID': df['ID'], 'Cluster': kmeans.labels_})
df_cluster.to_csv('Outputs\ID_Cluster.csv', index=False)

# 4.2) For each cluster, the coordinates of centroid in the PC1-PC2 space
centroids_pc = kmeans.cluster_centers_
df_centroids = pd.DataFrame(centroids_pc, columns=['PC1', 'PC2'])
df_centroids['Cluster'] = range(1, N_clusters + 1)
col = df_centroids.pop('Cluster')
df_centroids.insert(0, 'Cluster', col)
df_centroids.to_csv('Outputs\Centroids_PC.csv', index=False)