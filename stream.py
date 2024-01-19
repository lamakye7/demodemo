import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function for Isolation Forest anomaly detection
def isolation_forest_anomaly_detection(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_if = isolation_forest.fit_predict(X_scaled)

    return outliers_if

# Function to visualize anomalies
# Function to visualize anomalies
def visualize_anomalies(data, outliers, title):
    fig, ax = plt.subplots()

    # Plot inliers with label 'Inliers'
    ax.scatter(data[outliers == 0, 0], data[outliers == 0, 1], label='Inliers', cmap='viridis')

    # Plot outliers with label 'Outliers'
    ax.scatter(data[outliers == 1, 0], data[outliers == 1, 1], label='Outliers', cmap='viridis', marker='x')

    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


# Streamlit app
def main():
    st.title("Anomaly Detection App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Anomaly detection using Isolation Forest
        anomalies_if = isolation_forest_anomaly_detection(df.values)

        # Visualization
        st.subheader("Original Data")
        st.write(df.head())

        st.subheader("Anomaly Detection Visualization using Isolation Forest")
        visualize_anomalies(df.values, anomalies_if, "Isolation Forest Anomalies")

if __name__ == "__main__":
    main()

