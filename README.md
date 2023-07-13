# Clustering_traffic
A clustering analysis of road segments in terms of traffic patterns

This repo contains the (i) python code and (ii) dataset files concerning a clustering analysis of several Portuguese roads in terms of their traffic patterns. All the data is public. 

The dataset contains 3 features: TMDA, AMP, MAX, which have already been normalized by using the z-score.

The analysis starts with a PCA analysis, where the problem dimentionality is reduced to 2. They are related with traffic volume and seasonality.
The clustering analysis follows, and is performed with k-means method, using the euclidean distance as metric. 

After analysing the results, I selected the k=4 solution, which involves 4 clusters with the respective centroids sitting conveniently on each quadrant of the PC1-PC2 plane...
