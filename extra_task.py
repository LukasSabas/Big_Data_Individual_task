from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, to_date, col
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

# Spark setup
os.environ['PYSPARK_PYTHON'] = r'C:\Users\37068\AppData\Local\Programs\Python\Python39\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\37068\AppData\Local\Programs\Python\Python39\python.exe'

spark = SparkSession.builder.appName("LiquorSales").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load and preprocess data
def load_data(path):
    df_full = spark.read.csv(path, header=True, sep=",", quote="\"", escape="\"", multiLine=True)

    df = df_full.select(
        "Invoice/Item Number", "Date", "Store Number", "City", "County Number", "Vendor Number",
        "Item Number", "Pack", "Bottle Volume (ml)", "State Bottle Cost", "State Bottle Retail",
        "Bottles Sold", "Sale (Dollars)", "Volume Sold (Liters)"
    )

    df = df.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) \
        .withColumn("State Bottle Cost", regexp_replace("State Bottle Cost", "[$,]", "").cast("float")) \
        .withColumn("State Bottle Retail", regexp_replace("State Bottle Retail", "[$,]", "").cast("float")) \
        .withColumn("Sale (Dollars)", regexp_replace("Sale (Dollars)", "[$,]", "").cast("float")) \
        .withColumn("Store Number", col("Store Number").cast("int")) \
        .withColumn("County Number", col("County Number").cast("int")) \
        .withColumn("Vendor Number", col("Vendor Number").cast("int")) \
        .withColumn("Item Number", col("Item Number").cast("int")) \
        .withColumn("Pack", col("Pack").cast("int")) \
        .withColumn("Bottle Volume (ml)", col("Bottle Volume (ml)").cast("int")) \
        .withColumn("Bottles Sold", col("Bottles Sold").cast("int")) \
        .withColumn("Volume Sold (Liters)", col("Volume Sold (Liters)").cast("float"))

    return df

# PCA variance explained
def plot_pca_variance(df, features, max_components=10):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df_vector = assembler.transform(df).select("features")

    pca = PCA(k=max_components, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df_vector)

    explained_variance = model.explainedVariance.toArray()

    # Plot the explained variance
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.title('Explained Variance by PCA Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Determining the number of clisters
def compute_wss(df, features, pca_k=3, k_range=range(2, 11)):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df_vector = assembler.transform(df).select("features")

    pca = PCA(k=pca_k, inputCol="features", outputCol="pcaFeatures")
    pca_model = pca.fit(df_vector)
    df_pca = pca_model.transform(df_vector).select("pcaFeatures")

    wss_list = []
    for k in k_range:
        kmeans = KMeans(featuresCol="pcaFeatures", k=k, seed=42)
        model = kmeans.fit(df_pca)
        wss = model.summary.trainingCost
        print(f"k={k} -> WSS: {wss:.2f}")
        wss_list.append((k, wss))

    return wss_list
def plot_elbow(wss_list, title="Elbow Method for KMeans"):
    import matplotlib.pyplot as plt

    ks, wss = zip(*wss_list)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, wss, marker='o')
    plt.title(title)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to cluster with PCA and KMeans
def cluster_with_pca(df, features, k=3, pca_k=2, name='Model'):
    print(f"\nRunning {name}...")

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df_vector = assembler.transform(df).select(*features, "City", "features")

    pca = PCA(k=pca_k, inputCol="features", outputCol="pcaFeatures")
    pca_model = pca.fit(df_vector)
    df_pca = pca_model.transform(df_vector).select(*features, "City", "pcaFeatures")

    kmeans = KMeans(featuresCol="pcaFeatures", k=k, seed=42)
    kmeans_model = kmeans.fit(df_pca)
    clustered = kmeans_model.transform(df_pca)

    # Add prediction column as "Cluster"
    df_result = clustered.withColumnRenamed("prediction", "Cluster")

    # Convert to Pandas for visualization
    pd_data = df_result.toPandas()
    pd_data[['x', 'y']] = pd_data['pcaFeatures'].apply(lambda x: pd.Series(x.toArray()))

    return pd_data
# Visualization
def visualize_clusters_sample(data, name='Model', sample_size=100000):
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    plt.figure(figsize=(8, 6))
    for label in sorted(data['Cluster'].unique()):  # changed here
        subset = data[data['Cluster'] == label]  # and here
        plt.scatter(subset['x'], subset['y'], label=f'Cluster {label}', alpha=0.6, s=10)

    plt.title(f'{name} - PCA Clusters (Sample of {sample_size})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def visualize_clusters_density(data, name='Model'):
    plt.figure(figsize=(8, 6))
    plt.hexbin(data['x'], data['y'], gridsize=60, cmap='viridis', mincnt=1)
    plt.colorbar(label='Point Density')
    plt.title(f'{name} - PCA Cluster Density')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_clustered_cities_on_map(merged_df, state_center=[42.0, -93.0], zoom_start=7):
    # Create folium map
    m = folium.Map(location=state_center, zoom_start=zoom_start)

    # Prepare color map
    n_clusters = merged_df['Cluster'].nunique()
    colormap = cm.get_cmap('Set1', n_clusters)
    normalize = colors.Normalize(vmin=0, vmax=n_clusters - 1)

    # Add city markers
    for _, row in merged_df.iterrows():
        if pd.notnull(row['lat']) and pd.notnull(row['lng']):
            cluster_color = colors.rgb2hex(colormap(normalize(row['Cluster'])))
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=5,
                popup=f"{row['City']} (Cluster {row['Cluster']})",
                color=cluster_color,
                fill=True,
                fill_color=cluster_color,
                fill_opacity=0.9
            ).add_to(m)

    # Build the legend HTML string
    legend_html = '''
     <div style="
         position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: ''' + str(30 + 25 * n_clusters) + '''px; 
         border:2px solid grey; 
         background-color: white;
         z-index:9999;
         font-size:14px;
         padding: 10px;
         ">
     <b>Cluster Legend</b><br>
    '''
    # Add legend items
    for i in range(n_clusters):
        color = colors.rgb2hex(colormap(normalize(i)))
        legend_html += f'''<i style="background:{color};width:18px;height:18px;float:left;margin-right:8px;opacity:0.9;"></i>Cluster {i}<br>'''

    legend_html += '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))

    return m

####################### MAIN EXECUTION
path = "C:\\Users\\37068\\Desktop\\UNIVERSITETAS\\Magistras\\2 kursas\\Didžiųjų duomenų analizė\\Individual task\\Iowa_Liquor_Sales.csv" #Iowa_Liquor_Sales
df = load_data(path)

# Define meaningfull numeric features
features = [
    "Pack", "State Bottle Retail",
    "Bottles Sold", "Bottle Volume (ml)", "State Bottle Cost"
]

### PCA variance explained
plot_pca_variance(df.dropna(subset=features), features, 5) # 2

### Determining the number of clusters
wss_result1 = compute_wss(df.dropna(subset=features), features, pca_k=2, k_range=range(2, 8))
plot_elbow(wss_result1) # 5


### Run clustering and visualize
result = cluster_with_pca(df.dropna(subset=features), features, k=5, name="Model 1")

### Saving the result for faster use later
output_path = "C:\\Users\\37068\\Desktop\\UNIVERSITETAS\\Magistras\\2 kursas\\Didžiųjų duomenų analizė\\Individual task\\clustered_result.csv"
# result.to_csv(output_path, index=False)

# result = pd.read_csv(output_path)

###### RESULTS ANALYSIS

# # Visualize
visualize_clusters_sample(result, name="Model 1")
visualize_clusters_density(result, name="Model 1")

# # Tables
# Summary of each cluster by feature
cluster_summary = result.groupby('Cluster')[features].mean()
print("SUmmary of each cluster by feature")
print(cluster_summary)

# # Percentage of each cluster
cluster_percentages = result['Cluster'].value_counts(normalize=True) * 100
print("Percentage of each cluster size")
print(cluster_percentages.round(2))


# City analysis
# print(result["City"].unique())

output_path2 = "C:\\Users\\37068\\Desktop\\UNIVERSITETAS\\Magistras\\2 kursas\\Didžiųjų duomenų analizė\\Individual task\\city_coords.csv"
city_coords = pd.read_csv(output_path2)

result['City'] = result['City'].str.strip().str.title()
city_coords['name'] = city_coords['name'].str.strip().str.title()
merged_df = pd.merge(result, city_coords, left_on='City', right_on='name', how='left')


### Prep for vizualization
# Count number of rows for each (City, Cluster)
cluster_counts = merged_df.groupby(['City', 'Cluster']).size().reset_index(name='count')
# Sort descending so highest count is first
cluster_counts_sorted = cluster_counts.sort_values(['City', 'count'], ascending=[True, False])
# Keep only the row with the highest count per City
top_clusters = cluster_counts_sorted.drop_duplicates(subset=['City'], keep='first')
# Merge back with original merged_df to get 'City' and other columns
top_clusters = pd.merge(
    top_clusters,
    merged_df[['lat', 'lng', 'Cluster', 'City']],
    on=['Cluster', 'City'],
    how='left'
)

# Remove duplicate Cities (keep only one row per City)
top_clusters = top_clusters.drop_duplicates(subset=['City'])

iowa_map = plot_clustered_cities_on_map(top_clusters)
iowa_map.save("C:\\Users\\37068\\Desktop\\UNIVERSITETAS\\Magistras\\2 kursas\\Didžiųjų duomenų analizė\\Individual task\\clustered_iowa_cities.html")