from pydantic import BaseModel
from typing import List, Optional

class PredictionResultInput(BaseModel):
    provider_id: str
    prediction: str
    probability: float
    risk_factors: Optional[List[str]] = []

class ClaimData(BaseModel):
    provider_id: str
    features: list  # List of feature values in order
