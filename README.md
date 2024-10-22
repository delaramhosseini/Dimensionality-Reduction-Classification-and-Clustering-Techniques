# Dimensionality Reduction Classification and Clustering Techniques
Applying PCA for facial recognition and clustering analysis using K-Means, Fuzzy C-Means, and DBSCAN on different datasets.

## Part One: Facial Recognition with PCA and MLP Classifier
This project demonstrates facial recognition using Principal Component Analysis (PCA) and a Multi-Layer Perceptron (MLP) Classifier. The goal is to find the optimal number of PCA components that result in the highest accuracy for classifying faces in the LFW dataset. The project uses scikit-learn for data manipulation, dimensionality reduction, and machine learning tasks.

1. Load Data:

The project begins by loading the LFW (Labeled Faces in the Wild) dataset, which contains facial images.
Minimum faces per person and image resizing options are applied to filter the dataset.
The data is then stored in a Pandas DataFrame for further processing.

2. Principal Component Analysis (PCA):

Principal Component Analysis is applied to reduce the dimensionality of the data.
The pc function takes an argument a for the number of components to retain.
The function returns a DataFrame containing the data after PCA transformation.

3. Splitting Data:

The dataset is split into training and testing sets.
A random sample of 25% of the data is selected for testing, and the rest is used for training.

4. Model Training and Evaluation:

The MLPClassifier (Multi-Layer Perceptron) is used for face recognition.
The project iterates through different numbers of PCA components (from 10 to 79) to find the best accuracy.
For each number of components:
Data is transformed using PCA.
The transformed data is split into training and testing sets.
An MLP Classifier is trained on the training data.
Predictions are made on the testing data.
The accuracy of the model is calculated and printed.
The best accuracy and corresponding predictions are stored.

5. Best Component Selection:

After running the model for various numbers of PCA components, the project identifies the best count of components with the highest accuracy.

6. Confusion Matrix:

Finally, a confusion matrix is generated to evaluate the model's performance.
The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives.

## Part Tow: Clustering Analysis with K-Means, Fuzzy C-Means, and DBSCAN
This project explores clustering analysis using different techniques such as K-Means, Fuzzy C-Means, and DBSCAN on three distinct datasets. The goal is to discover meaningful patterns and groupings within the data. The project utilizes Python libraries like pandas, matplotlib, scikit-learn, numpy, and the fcmeans package.
The project begins by loading three different datasets:

"first_clustering_dataset.csv"
"second_clustering_dataset.csv"
"third_clustering_dataset.csv"
Each dataset contains two-dimensional data points.

### K-Means Clustering

The project utilizes K-Means clustering to group data points into clusters. The KMEANS function is defined to perform K-Means clustering with the following steps:

Initialize a K-Means model with a specified number of clusters (number_of_clusters) and random initialization repetitions (random_c).
Fit the model to the data.
Visualize the clustered data points along with centroids.
Calculate the Sum of Squared Errors (SSE) and silhouette score for evaluating clustering quality.
The KMEANS function is applied to each dataset with different cluster counts:

For "first_clustering_dataset" (K=2):

SSE: 146.90
Silhouette Score: 0.536
For "second_clustering_dataset" (K=3):

SSE: 6481.46
Silhouette Score: 0.723
For "third_clustering_dataset" (K=5):

SSE: 222.20
Silhouette Score: 0.560

### Fuzzy C-Means Clustering

Fuzzy C-Means (FCM) clustering is applied to the datasets to allow data points to belong to multiple clusters with varying degrees of membership. The fcm_func function is defined to perform FCM clustering with the following steps:

Initialize an FCM model with a specified number of clusters (number_of_cluster).
Fit the model to the data.
Visualize the clustered data points along with cluster centers.
Calculate the SSE and silhouette score for evaluating clustering quality.
The fcm_func function is applied to each dataset with different cluster counts:

For "first_clustering_dataset" (K=2):

SSE: 146.97
Silhouette Score: 0.536
For "second_clustering_dataset" (K=3):

SSE: 6483.79
Silhouette Score: 0.723
For "third_clustering_dataset" (K=5):

SSE: 222.70
Silhouette Score: 0.560

### DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering is applied to the datasets. DBSCAN does not require specifying the number of clusters in advance but identifies clusters based on data density. The dbscan_func function is defined to perform DBSCAN clustering with the following steps:

Initialize a DBSCAN model with specified parameters (v_eps for epsilon and v_min_samples for minimum samples).
Fit the model to the data.
Visualize the clustered data points.
The dbscan_func function is applied to each dataset with different parameters:

For "first_clustering_dataset" (eps=0.25, min_samples=10):

Silhouette Score: 0.521
For "second_clustering_dataset" (eps=6, min_samples=2):

Silhouette Score: 0.723
For "third_clustering_dataset" (eps=0.39, min_samples=10):

Silhouette Score: 0.510
