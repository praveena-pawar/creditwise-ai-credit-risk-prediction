from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()


model = joblib.load("../models/xgboost_model.pkl")

class LoanApplication(BaseModel):
    applicant_income: float
    coapplicant_income: float
    age: int
    dependents: int
    credit_score: int
    existing_loans: int
    dti_ratio: float
    savings: float
    collateral_value: float
    loan_amount: float
    loan_term: int

    education_level: str
    employment_status: str
    marital_status: str
    loan_purpose: str
    property_area: str
    gender: str
    employer_category: str




@app.get("/")
def home():
    return {"message": "Welcome to CreditWise API"}



@app.post("/predict")
def predict(data: LoanApplication):

    features = {
        "Applicant_Income": data.applicant_income,
        "Coapplicant_Income": data.coapplicant_income,
        "Age": data.age,
        "Dependents": data.dependents,
        "Credit_Score": data.credit_score,
        "Existing_Loans": data.existing_loans,
        "DTI_Ratio": data.dti_ratio,
        "Savings": data.savings,
        "Collateral_Value": data.collateral_value,
        "Loan_Amount": data.loan_amount,
        "Loan_Term": data.loan_term,

        # Label Encoded
        "Education_Level": 0 if data.education_level == "Graduate" else 1,

        # One-Hot Encoded
        "Employment_Status_Salaried": 0,
        "Employment_Status_Self-employed": 0,
        "Employment_Status_Unemployed": 0,

        "Marital_Status_Single": 0,

        "Loan_Purpose_Car": 0,
        "Loan_Purpose_Education": 0,
        "Loan_Purpose_Home": 0,
        "Loan_Purpose_Personal": 0,

        "Property_Area_Semiurban": 0,
        "Property_Area_Urban": 0,

        "Gender_Male": 0,

        "Employer_Category_Government": 0,
        "Employer_Category_MNC": 0,
        "Employer_Category_Private": 0,
        "Employer_Category_Unemployed": 0,
    }


    #Employment Status
    if data.employment_status == "Salaried":
        features["Employment_Status_Salaried"] = 1
    elif data.employment_status == "Self-employed":
        features["Employment_Status_Self-employed"] = 1
    elif data.employment_status == "Unemployed":
        features["Employment_Status_Unemployed"] = 1


    #Marital Status
    if data.marital_status == "Single":
        features["Marital_Status_Single"] = 1


    #Loan Purpose
    if data.loan_purpose == "Car":
        features["Loan_Purpose_Car"] = 1
    elif data.loan_purpose == "Education":
        features["Loan_Purpose_Education"] = 1
    elif data.loan_purpose == "Home":
        features["Loan_Purpose_Home"] = 1
    elif data.loan_purpose == "Personal":
        features["Loan_Purpose_Personal"] = 1


    #Property Area
    if data.property_area == "Semiurban":
        features["Property_Area_Semiurban"] = 1
    elif data.property_area == "Urban":
        features["Property_Area_Urban"] = 1


    # Gender
    if data.gender == "Male":
        features["Gender_Male"] = 1


    # Employer Category
    if data.employer_category == "Government":
        features["Employer_Category_Government"] = 1
    elif data.employer_category == "MNC":
        features["Employer_Category_MNC"] = 1
    elif data.employer_category == "Private":
        features["Employer_Category_Private"] = 1
    elif data.employer_category == "Unemployed":
        features["Employer_Category_Unemployed"] = 1


    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    confidence = float(round(max(probability[0]) * 100, 2))

    loan_status = "Approved" if prediction[0] == 1 else "Rejected"

    return {
        "loan_status": loan_status,
        "confidence": f"{confidence:.2f}%"
    }
