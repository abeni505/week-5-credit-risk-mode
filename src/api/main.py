from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
# Make sure pydantic_models.py is in the same directory (src/api/)
from .pydantic_models import CreditRiskInput, PredictionResponse
import os

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk based on customer transaction data."
)

# --- Model Loading (Updated for Robustness) ---
MODEL_NAME = "CreditRiskModel"
# We will try to load from Production first, then fall back to Staging, then None.
STAGES_TO_TRY = ["Production", "Staging", "None"] 

model = None
for stage in STAGES_TO_TRY:
    try:
        print(f"Attempting to load model '{MODEL_NAME}' from stage '{stage}'...")
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        # Load the scikit-learn flavor of the model to access predict_proba
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Successfully loaded model from stage: '{stage}'")
        break # Exit the loop if model is loaded successfully
    except Exception as e:
        print(f"Could not load model from stage '{stage}'. Error: {e}")

if model is None:
    print("FATAL: Could not load the model from any stage. The API will not be able to serve predictions.")


@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "API is running successfully"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CreditRiskInput):
    """
    Accepts new customer data and returns the risk probability.
    
    The input features must match the model's training data exactly.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please check the application logs."
        )

    try:
        # Convert the Pydantic model to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Use predict_proba to get the probability of each class
        # It returns an array like [[prob_class_0, prob_class_1]]
        probabilities = model.predict_proba(input_df)
        
        # We want the probability of the positive class (is_high_risk = 1)
        risk_probability = probabilities[0][1]
        
        # Determine the class based on a 0.5 threshold
        is_high_risk = 1 if risk_probability > 0.5 else 0

        return PredictionResponse(
            risk_probability=risk_probability,
            is_high_risk=is_high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
