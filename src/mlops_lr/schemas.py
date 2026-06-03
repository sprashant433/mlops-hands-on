from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=21, le=65)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0)
    debt_to_income: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    loan_approved: int
    probability: float
