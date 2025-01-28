import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load pre-trained models
kmeans = pickle.load(open('model/kmeans.pkl', 'rb'))

# Application Title
st.title("Customer Segmentation and Investment Recommendation")

# File Uploader
uploaded_file = st.file_uploader("Upload a customer data CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Show first few rows
    st.write("Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing (you can expand this step)
    features = ["Age", "Income", "Account_Balance", "Monthly_Spending", "Credit_Score"]
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict Clusters
    cluster_labels = kmeans.predict(X_scaled)
    data['Cluster'] = cluster_labels

    # Recommendation Logic (add based on your earlier logic)
    def recommend(row):
        if row['Cluster'] == 0:
            return "Premium Bonds"
        elif row['Cluster'] == 1:
            return "Growth Stocks"
        elif row['Cluster'] == 2:
            return "Fixed Deposit"
        elif row['Cluster'] == 3:
            return "Index Funds"
        else:
            return "Consultation Required"

    # Apply recommendations
    data['Recommendation'] = data.apply(recommend, axis=1)

    # Display Results
    st.write("Customer Segmentation Results")
    st.dataframe(data)

    # Download Processed Results
    csv = data.to_csv(index=False)
    st.download_button("Download Results as CSV", data=csv, file_name="Segmented_Customers.csv", mime="text/csv")
