from pydantic import BaseModel
from typing import List

# NOTE: The feature names here must exactly match the columns 
# in the data used to train the model.
# This is based on the output of our data_processing.py script.

class CreditRiskInput(BaseModel):
    """
    Input features for a single credit risk prediction.
    """
    total_transactions: int
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_min: float
    Amount_max: float
    ProductId_nunique: int
    TransactionHour_mean: float
    TransactionHour_std: float
    most_frequent_day: int # e.g., 0 for Monday, 6 for Sunday

class PredictionResponse(BaseModel):
    """
    Prediction response schema.
    """
    risk_probability: float
    is_high_risk: int # 0 for low risk, 1 for high risk
