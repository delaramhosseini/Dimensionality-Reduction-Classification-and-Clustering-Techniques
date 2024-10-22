# Dimensionality Reduction Classification and Clustering Techniques
This project applies Principal Component Analysis (PCA) for facial recognition and utilizes various clustering algorithms—K-Means, Fuzzy C-Means, and DBSCAN—to analyze different datasets.

## Part One: Facial Recognition with PCA and MLP Classifier
This section demonstrates facial recognition through the use of **Principal Component Analysis (PCA)** and a **Multi-Layer Perceptron (MLP) Classifier**. The primary objective is to determine the optimal number of PCA components that yield the highest classification accuracy for faces in the **LFW (Labeled Faces in the Wild)** dataset. The project leverages the **scikit-learn** library for data manipulation, dimensionality reduction, and machine learning tasks.

### 1. Load Data
The project begins by loading the LFW dataset, which consists of images of faces. The following steps are performed:
- Filters are applied based on the minimum number of faces per person and image resizing options to enhance data quality.
- The filtered data is organized and stored in a **Pandas DataFrame** for further analysis.

### 2. Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction while preserving as much information as possible in the dataset. PCA is a powerful tool for simplifying datasets, making them easier to visualize and analyze while retaining the most critical information.
- The `pc` function is utilized, where an argument `a` specifies the number of components to retain.
- This function outputs a new DataFrame containing the transformed data post-PCA.

### 3. Splitting Data
The dataset is divided into training and testing sets to evaluate model performance. 
- A random sample of 25% of the data is designated for testing, while the remaining 75% is used for training the model.

### 4. Model Training and Evaluation
The **MLPClassifier**, a type of neural network, is employed for face recognition. The project iterates through various PCA component counts (from 10 to 79) to identify the configuration that produces the best accuracy. For each count of components:
- Data is transformed using PCA.
- The transformed data is split into training and testing sets.
- The MLP Classifier is trained on the training dataset.
- Predictions are made on the testing dataset.
- Model accuracy is calculated and displayed. 
- The highest accuracy and its corresponding configuration are recorded for later analysis.

### 5. Best Component Selection
After evaluating multiple PCA configurations, the project identifies the optimal number of components that yields the highest classification accuracy.

### 6. Confusion Matrix
A confusion matrix is generated to visualize the model's performance. This matrix displays the number of:
- True Positives (correctly predicted positive cases)
- True Negatives (correctly predicted negative cases)
- False Positives (incorrectly predicted as positive)
- False Negatives (incorrectly predicted as negative)

## Part Two: Clustering Analysis with K-Means, Fuzzy C-Means, and DBSCAN
This section investigates clustering techniques using K-Means, Fuzzy C-Means, and **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on three distinct datasets. The goal is to identify meaningful patterns and groupings within the data. The project utilizes libraries such as **pandas**, **matplotlib**, **scikit-learn**, **numpy**, and the **fcmeans** package for fuzzy clustering.

The datasets analyzed include:
- **first_clustering_dataset.csv**
- **second_clustering_dataset.csv**
- **third_clustering_dataset.csv**
Each dataset comprises two-dimensional data points.

### K-Means Clustering
K-Means clustering is utilized to partition data points into distinct clusters. The `KMEANS` function is defined with the following steps:
- Initialize a K-Means model with a specified number of clusters and random initialization repetitions.
- Fit the model to the data.
- Visualize the clustered data points alongside cluster centroids.
- Compute the **Sum of Squared Errors (SSE)** and **silhouette score** to evaluate clustering quality.

The `KMEANS` function is applied to each dataset with varying cluster counts:
- For **first_clustering_dataset** (K=2): 
  - SSE: 146.90
  - Silhouette Score: 0.536
- For **second_clustering_dataset** (K=3):
  - SSE: 6481.46
  - Silhouette Score: 0.723
- For **third_clustering_dataset** (K=5):
  - SSE: 222.20
  - Silhouette Score: 0.560

### Fuzzy C-Means Clustering
**Fuzzy C-Means (FCM)** clustering is employed to allow data points to belong to multiple clusters with varying membership degrees. The `fcm_func` function is defined with the following steps:
- Initialize an FCM model with a specified number of clusters.
- Fit the model to the data.
- Visualize the clustered data points along with the cluster centers.
- Calculate SSE and silhouette scores for evaluating clustering quality.

The `fcm_func` function is applied to each dataset:
- For **first_clustering_dataset** (K=2):
  - SSE: 146.97
  - Silhouette Score: 0.536
- For **second_clustering_dataset** (K=3):
  - SSE: 6483.79
  - Silhouette Score: 0.723
- For **third_clustering_dataset** (K=5):
  - SSE: 222.70
  - Silhouette Score: 0.560

### DBSCAN Clustering
DBSCAN clustering is applied to the datasets, which does not require pre-specifying the number of clusters; instead, it identifies clusters based on data density. The `dbscan_func` function is defined with these steps:
- Initialize a DBSCAN model with specified parameters (`eps` for neighborhood radius and `min_samples` for minimum data points in a neighborhood).
- Fit the model to the data.
- Visualize the clustered data points.

The `dbscan_func` is applied to each dataset with varying parameters:
- For **first_clustering_dataset** (eps=0.25, min_samples=10):
  - Silhouette Score: 0.521
- For **second_clustering_dataset** (eps=6, min_samples=2):
  - Silhouette Score: 0.723
- For **third_clustering_dataset** (eps=0.39, min_samples=10):
  - Silhouette Score: 0.510
