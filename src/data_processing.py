import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(file_path):
    """Loads the raw data from a CSV file."""
    print("Loading data...")
    try:
        df = pd.read_csv(file_path)
        # Drop the 'Value' column as it's perfectly correlated with 'Amount'
        if 'Value' in df.columns:
            df = df.drop(columns=['Value'])
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

def extract_time_features(df):
    """Extracts time-based features from the TransactionStartTime column."""
    print("Extracting time-based features...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek # Monday=0, Sunday=6
    return df

def create_aggregate_features(df):
    """Performs feature engineering by creating aggregate features for each customer."""
    print("Engineering aggregate features...")
    
    # Define aggregation logic
    agg_logic = {
        'TransactionId': ['count'],
        'Amount': ['sum', 'mean', 'std', 'min', 'max'],
        'ProductId': ['nunique'],
        'TransactionHour': ['mean', 'std'],
        'TransactionDayOfWeek': [lambda x: x.mode()[0]] # Mode for day of week
    }
    
    customer_agg = df.groupby('CustomerId').agg(agg_logic)
    
    # Flatten the multi-level column names
    customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
    customer_agg.rename(columns={'TransactionId_count': 'total_transactions',
                                 'TransactionDayOfWeek_<lambda>': 'most_frequent_day'}, inplace=True)
    
    # Fill NaN in std columns for customers with only one transaction
    std_cols = [col for col in customer_agg.columns if 'std' in col]
    for col in std_cols:
        customer_agg[col] = customer_agg[col].fillna(0)
        
    return customer_agg.reset_index()


def main():
    """Main function to run the feature engineering pipeline."""
    raw_data_path = 'data/raw/training.csv'
    features_output_path = 'data/processed/customer_features.csv'
    os.makedirs('data/processed', exist_ok=True)

    raw_df = load_data(raw_data_path)
    if raw_df is None:
        return

    # --- Task 3 Pipeline ---
    # 1. Extract time-based features from raw data
    time_featured_df = extract_time_features(raw_df.copy())
    
    # 2. Create aggregate features for each customer
    customer_features_df = create_aggregate_features(time_featured_df)
    
    # Isolate features for the pipeline
    X = customer_features_df.drop(columns=['CustomerId'])

    # --- Sklearn Pipeline for Preprocessing ---
    print("\nBuilding and applying preprocessing pipeline...")
    
    # Identify column types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['most_frequent_day'] 
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    final_feature_names = numerical_features + ohe_feature_names.tolist()

    # Create the final processed DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=final_feature_names, index=X.index)

    # Combine processed features with CustomerId
    final_df = pd.concat([customer_features_df[['CustomerId']], X_processed_df], axis=1)

    print("\nFinal processed and scaled features dataframe head:")
    print(final_df.head())
    
    # Save the processed features
    final_df.to_csv(features_output_path, index=False)
    print(f"\nProcessed features saved successfully to {features_output_path}")

if __name__ == '__main__':
    main()
