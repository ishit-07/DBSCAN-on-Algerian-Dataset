import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Define the file path to the dataset
file_path = r'C:\Users\aashish\OneDrive\Documents\DBSCAN-Algorithm-master\Algerian_forest_fires_dataset_UPDATE.csv'

try:
    # Load the dataset
    data = pd.read_csv(file_path)

    # Print the first few rows of the loaded data for inspection
    print(data.head())  # Print the first few rows

    # Print the column names
    print(data.columns)  # Print the column names

    # Preprocess text data and handle missing values
    data['Bejaia Region Dataset '] = data['Bejaia Region Dataset '].fillna('')  # Replace NaN with an empty string

    # Apply TF-IDF to convert text data to numerical vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Bejaia Region Dataset '])

    # Perform DBSCAN clustering
    eps = 0.3  # Adjust the epsilon value as needed
    min_samples = 5  # Adjust the minimum samples as needed
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(tfidf_matrix)

    # Get the predicted labels
    labels = dbscan.labels_

    # Evaluate the clustering results
    silhouette = silhouette_score(tfidf_matrix, labels)

    # Print evaluation metrics
    print("Silhouette Score:", silhouette)

    # Visualize the clusters (if applicable)
    # Note: Visualization of text data clustering can be complex

except FileNotFoundError:
    print(f"File not found at the specified path: {file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# After DBSCAN clustering, you have 'labels' which represent cluster assignments.
# You can assign these labels to your data points as a new column.
data['cluster_labels'] = dbscan.labels_

# Now, each data point is assigned to a cluster, and you can analyze or visualize the clusters.
print(data.head())
