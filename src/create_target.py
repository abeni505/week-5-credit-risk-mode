import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

def calculate_rfm(df):
    """Calculates Recency, Frequency, and Monetary (RFM) metrics for each customer."""
    print("Calculating RFM metrics...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })
    
    rfm.rename(columns={'TransactionStartTime': 'Recency',
                        'TransactionId': 'Frequency',
                        'Amount': 'Monetary'}, inplace=True)
                        
    return rfm.reset_index()

def create_risk_proxy(rfm_df):
    """Uses K-Means clustering on RFM data to create a 'is_high_risk' proxy variable."""
    print("Creating high-risk proxy variable using K-Means clustering...")
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    cluster_analysis = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\nRFM Cluster Analysis (Centroids):")
    print(cluster_analysis)
    
    # High-risk is typically high recency (inactive), low frequency, and low monetary value.
    high_risk_cluster = cluster_analysis['Recency'].idxmax()
    print(f"\nIdentified High-Risk Cluster: {high_risk_cluster}")
    
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm_df[['CustomerId', 'is_high_risk']]

def main():
    """Main function to run the target engineering pipeline."""
    raw_data_path = 'data/raw/training.csv'
    features_path = 'data/processed/customer_features.csv'
    final_output_path = 'data/processed/final_training_data.csv'
    
    # Load raw data for RFM and engineered features
    raw_df = load_data(raw_data_path)
    features_df = load_data(features_path)
    
    if raw_df is None or features_df is None:
        print("Could not proceed due to missing files.")
        return

    # --- Task 4 Pipeline ---
    # 1. Calculate RFM metrics
    rfm_df = calculate_rfm(raw_df.copy())
    
    # 2. Create the proxy target variable
    risk_proxy_df = create_risk_proxy(rfm_df)
    
    # 3. Merge features and target
    final_df = pd.merge(features_df, risk_proxy_df, on='CustomerId')
    
    print("\nFinal training dataframe with features and target head:")
    print(final_df.head())
    
    # Save the final training data
    final_df.to_csv(final_output_path, index=False)
    print(f"\nFinal training data saved successfully to {final_output_path}")

if __name__ == '__main__':
    main()
