# Big_Data_Individual_task
Individual assignment of the final taks for Big Data

Description:
This project analyzes Iowa liquor sales data using PySpark for data processing and MLlib for machine learning.
PCA is used for dimensionality reduction, followed by KMeans clustering to segment stores or products based on numerical features.
The results are visualized using matplotlib and folium.


1. Function Descriptions:
load_data(path)
Loads CSV data, selects relevant columns, and performs type conversion and cleaning.

plot_pca_variance(df, features, max_components)
Applies PCA and plots the variance explained by each principal component.

compute_wss(df, features, pca_k, k_range)
Computes Within-Cluster Sum of Squares (WSS) for a range of k values using PCA + KMeans.

plot_elbow(wss_list, title)
Plots the WSS results from compute_wss to find the elbow point (optimal cluster number).

cluster_with_pca(df, features, k, pca_k, name)
Performs PCA and KMeans clustering, returns the labeled data as a Pandas DataFrame.

visualize_clusters_sample(data, name, sample_size)
Visualizes sampled clustered data using scatter plot by cluster.

visualize_clusters_density(data, name)
Plots hexbin density of clusters based on PCA components.

plot_clustered_cities_on_map(merged_df, state_center, zoom_start)
Visualizes cities and clusters on a Folium map using latitude and longitude coordinates.

2. Step-by-Step Process:
Start Spark session
Configure Python executable and initialize PySpark session.

Load and preprocess CSV data
Loads the dataset and converts string columns to proper numeric types.
Define numerical features
Selects the most meaningful features for PCA and clustering.

Visualize PCA variance
plot_pca_variance(...)
Determines how many principal components to retain based on explained variance.

Determine number of clusters
wss_result1 = compute_wss(...)
plot_elbow(wss_result1)
Calculates WSS for k=2 to k=7 and visualizes the elbow curve.

Perform clustering
result = cluster_with_pca(...)
Applies PCA and KMeans to assign each row a cluster label.

Visualize clusters in PCA space
visualize_clusters_sample(result)
visualize_clusters_density(result)
Two visualizations: scatter and hexbin for cluster shapes.

Analyze clustering results
Print mean of features per cluster
Print percentage of records per cluster

City coordinate processing
Read in city_coords and clean City names to allow joining.

Join cluster results with city coordinates
Merge on City name to get latitude/longitude.

Select dominant cluster per city
Count how many records per city/cluster and choose the most common.

Visualize clusters on map
plot_clustered_cities_on_map(...)
Saves a .html file with city markers colored by cluster.

