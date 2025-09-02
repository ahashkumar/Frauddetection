from sqlalchemy import Column, Integer, String, Float, Text
from database import Base

class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String, index=True)
    prediction = Column(String)
    probability = Column(Float)
    risk_factors = Column(Text, nullable=True)
