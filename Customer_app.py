import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load pre-trained models
kmeans = pickle.load(open('model/kmeans.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# Application Title
st.title("Customer Segmentation and Investment Recommendation System")

# Instructions
st.markdown("""
### Instructions:
1. Upload a CSV file containing customer data.
2. Ensure the dataset contains the following features:
   - **Age**, **Income**, **Account_Balance**, **Monthly_Spending**, **Credit_Score**.
3. View segmentation results and investment recommendations.
4. Download the processed dataset with recommendations.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload a customer data CSV", type=["csv"])
if uploaded_file:
    # Load uploaded data
    data = pd.read_csv(uploaded_file)
    
    # Show first few rows of the dataset
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        # Preprocessing: Selecting relevant columns and scaling
        features = ["Age", "Income", "Account_Balance", "Monthly_Spending", "Credit_Score"]
        X = data[features]
        X_scaled = scaler.transform(X)

        # Predict Clusters
        cluster_labels = kmeans.predict(X_scaled)
        data['Cluster'] = cluster_labels

        # Define Recommendation Logic
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

        # Show Results
        st.write("### Segmentation Results with Recommendations")
        st.dataframe(data)

        # Download Processed Data
        csv = data.to_csv(index=False)
        st.download_button("Download Results as CSV", data=csv, file_name="Segmented_Customers.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
else:
    st.info("Upload a file to proceed.")

