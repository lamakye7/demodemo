import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    st.title("Anomaly Detection App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Process data
        X = df.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Isolation Forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest.fit(X_scaled)
        outliers_if = isolation_forest.predict(X_scaled)

        # One-Class SVM
        one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        one_class_svm.fit(X_scaled)
        outliers_svm = one_class_svm.predict(X_scaled)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(X_scaled)
        distances = np.sum((X_scaled - kmeans.cluster_centers_)**2, axis=1)
        threshold_kmeans = np.percentile(distances, 95)
        outliers_kmeans = (distances > threshold_kmeans).astype(int)

        # Visualize the anomalies
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.scatter(X[:, 0], X[:, 1], c=outliers_if, cmap='viridis')
        plt.title('Isolation Forest Anomalies')

        plt.subplot(132)
        plt.scatter(X[:, 0], X[:, 1], c=outliers_svm, cmap='viridis')
        plt.title('One-Class SVM Anomalies')

        plt.subplot(133)
        plt.scatter(X[:, 0], X[:, 1], c=outliers_kmeans, cmap='viridis')
        plt.title('K-Means Clustering Anomalies')

        st.pyplot()

def load_data(file):
    # Load data from CSV or Excel file
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format")

    return df

if __name__ == "__main__":
    main()



   

