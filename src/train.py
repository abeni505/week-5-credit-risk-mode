import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(file_path):
    """Loads the final processed data."""
    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

def eval_metrics(actual, pred, pred_proba):
    """Calculates and returns a dictionary of model evaluation metrics."""
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred_proba)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def main():
    """Main function to run the model training and tracking pipeline."""
    print("--- Starting Model Training Script ---")
    # --- 1. Setup ---
    data_path = 'data/processed/final_training_data.csv'
    experiment_name = "Credit_Risk_Modeling"
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # --- 2. Load and Split Data ---
    df = load_data(data_path)
    if df is None:
        print("\nData loading failed. Please ensure you have run the previous data processing scripts successfully.")
        print("Exiting script.")
        return
    else:
        print("Data loaded successfully. Proceeding with training.")
        
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 3. Model Training and Hyperparameter Tuning ---
    
    # --- Logistic Regression ---
    with mlflow.start_run(run_name="LogisticRegression_GridSearch"):
        print("\n--- Training Logistic Regression ---")
        
        param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
        lr = LogisticRegression(random_state=42)
        grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search_lr.fit(X_train, y_train)
        best_lr = grid_search_lr.best_estimator_
        
        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        
        # --- FIX: Add input_example to infer signature ---
        signature = infer_signature(X_train, best_lr.predict(X_train))
        
        mlflow.log_params(grid_search_lr.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=best_lr, 
            artifact_path="logistic_regression_model", # Using artifact_path for clarity
            signature=signature,
            input_example=X_train.head(5)
        )
        
        print("Logistic Regression Metrics:", metrics)

    # --- Random Forest ---
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        print("\n--- Training Random Forest ---")
        
        param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        best_rf = grid_search_rf.best_estimator_
        
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)

        # --- FIX: Add input_example to infer signature ---
        signature = infer_signature(X_train, best_rf.predict(X_train))
        
        mlflow.log_params(grid_search_rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=best_rf, 
            artifact_path="random_forest_model", # Using artifact_path for clarity
            signature=signature,
            input_example=X_train.head(5)
        )

        print("Random Forest Metrics:", metrics)

    # --- 4. Identify and Register Best Model ---
    print("\n--- Identifying and Registering Best Model ---")
    
    best_run = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    ).iloc[0]
    
    run_name = best_run["tags.mlflow.runName"]
    if "RandomForest" in run_name:
        model_artifact_path = "random_forest_model"
    elif "LogisticRegression" in run_name:
        model_artifact_path = "logistic_regression_model"
    else:
        raise ValueError(f"Could not determine model type from run name: {run_name}")

    best_model_uri = f"runs:/{best_run.run_id}/{model_artifact_path}"
    model_name = "CreditRiskModel"
    
    print(f"Best model found in run: {best_run.run_id} ({run_name})")
    print(f"Best model ROC AUC: {best_run['metrics.roc_auc']:.4f}")
    print(f"Registering model '{model_name}' from {best_model_uri}")

    registered_model = mlflow.register_model(
        model_uri=best_model_uri,
        name=model_name
    )
    
    print(f"Model '{registered_model.name}' version {registered_model.version} registered successfully.")
    
    print("\n--- Model Training Script Finished ---")

if __name__ == '__main__':
    main()
