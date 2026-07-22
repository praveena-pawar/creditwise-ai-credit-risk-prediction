import streamlit as st
import pandas as pd
import joblib

with st.sidebar:
    st.header("🏦 CreditWise")

    st.markdown("### 📌 About")
    st.write(
        """
        AI-powered Credit Risk Prediction System
        that predicts whether a loan application
        is likely to be approved or rejected.
        """
    )

    st.divider()

    st.markdown("### 🤖 Model")
    st.write("**XGBoost Classifier**")

    st.markdown("### 🎯 Accuracy")
    st.write("**92.5%**")

    st.markdown("### 🛠 Tech Stack")
    st.write("""
- Python
- Pandas
- XGBoost
- Scikit-learn
- Streamlit
""")


st.set_page_config(
    page_title="CreditWise",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load trained model
model = joblib.load("models/xgboost_model.pkl")



# -------------------------------
# Title
# -------------------------------
st.title("🏦 CreditWise")
st.subheader("AI-Powered Credit Risk Prediction System")

st.markdown(
    """
Predict whether a loan application is likely to be **Approved** or **Rejected**
using a Machine Learning model trained on applicant financial information.
"""
)

st.divider()



# -------------------------------
# Numeric Inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Information")

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=2009,
        max_value=19988,
        value=10000,
        help="Monthly applicant income"
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=1,
        max_value=9996,
        value=5000
    )

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=59,
        value=30
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=3,
        value=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=550,
        max_value=799,
        value=700
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


with col2:
    st.subheader("🏦 Loan Information")

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=4,
        value=1
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.10,
        max_value=0.60,
        value=0.30,
        step=0.01
    )

    savings = st.number_input(
        "Savings",
        min_value=65,
        max_value=19996,
        value=10000
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=36,
        max_value=49954,
        value=25000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1015,
        max_value=39995,
        value=15000
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=12,
        max_value=360,
        value=120
    )



# -------------------------------
# Dropdown Inputs
# -------------------------------

employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Car", "Education", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)


employer_category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)



if st.button("🔍 Predict Loan Eligibility", use_container_width=True):

    # ---------------------------------
    # Create Feature Dictionary
    # ---------------------------------
    features = {
        # Numerical Features
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,

        # Label Encoding
        "Education_Level": 0 if education_level == "Graduate" else 1,

        # One-Hot Encoding
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
        "Employer_Category_Unemployed": 0
    }



    # ---------------------------------
    # One-Hot Encoding
    # ---------------------------------

    if employment_status == "Salaried":
        features["Employment_Status_Salaried"] = 1
    elif employment_status == "Self-employed":
        features["Employment_Status_Self-employed"] = 1
    else:
        features["Employment_Status_Unemployed"] = 1

    if marital_status == "Single":
        features["Marital_Status_Single"] = 1

    if loan_purpose == "Car":
        features["Loan_Purpose_Car"] = 1
    elif loan_purpose == "Education":
        features["Loan_Purpose_Education"] = 1
    elif loan_purpose == "Home":
        features["Loan_Purpose_Home"] = 1
    elif loan_purpose == "Personal":
        features["Loan_Purpose_Personal"] = 1

    if property_area == "Semiurban":
        features["Property_Area_Semiurban"] = 1
    elif property_area == "Urban":
        features["Property_Area_Urban"] = 1

    if gender == "Male":
        features["Gender_Male"] = 1

    if employer_category == "Government":
        features["Employer_Category_Government"] = 1
    elif employer_category == "MNC":
        features["Employer_Category_MNC"] = 1
    elif employer_category == "Private":
        features["Employer_Category_Private"] = 1
    elif employer_category == "Unemployed":
        features["Employer_Category_Unemployed"] = 1



    # ---------------------------------
    # Create DataFrame
    # ---------------------------------

    input_df = pd.DataFrame([features])



    # ---------------------------------
    # Input Validation
    # ---------------------------------

    if loan_amount > collateral_value:
        st.warning("⚠️ Loan amount is greater than the collateral value.")

    if credit_score < 600:
        st.warning("⚠️ Low credit score may reduce approval chances.")

    if dti_ratio > 0.45:
        st.warning("⚠️ High Debt-to-Income ratio.")

    if existing_loans >= 3:
        st.warning("⚠️ Applicant already has multiple existing loans.")

    if savings < 2000:
        st.warning("⚠️ Low savings balance.")



    # ---------------------------------
    # Prediction
    # ---------------------------------

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    confidence = float(round(max(probability[0]) * 100, 2))

    loan_status = "Approved" if prediction[0] == 1 else "Rejected"



    # ---------------------------------
    # Display Result
    # ---------------------------------

    if loan_status == "Approved":
        st.balloons()
        st.success("✅ Congratulations! Loan Approved")

        st.markdown("### 📋 Why this prediction?")

        reasons = []

        if credit_score >= 750:
            reasons.append("✅ Excellent credit score")

        if dti_ratio <= 0.30:
            reasons.append("✅ Healthy Debt-to-Income Ratio")

        if existing_loans == 0:
            reasons.append("✅ No existing loans")

        if savings >= 10000:
            reasons.append("✅ Strong savings balance")

        if collateral_value >= loan_amount:
            reasons.append("✅ Sufficient collateral")

        for reason in reasons:
            st.write(reason)

    else:

        st.error("❌ Loan Rejected")

        st.markdown("### ⚠️ Possible Reasons")

        reasons = []

        if credit_score < 650:
            reasons.append("❌ Low credit score")

        if dti_ratio > 0.45:
            reasons.append("❌ High Debt-to-Income Ratio")

        if existing_loans >= 3:
            reasons.append("❌ Too many existing loans")

        if savings < 5000:
            reasons.append("❌ Low savings")

        if loan_amount > collateral_value:
            reasons.append("❌ Loan amount exceeds collateral value")

        if len(reasons) == 0:
            reasons.append("❌ The model found a combination of factors that increased the predicted risk.")

        for reason in reasons:
            st.write(reason)


    # ---------------------------------
    # Confidence Score
    # ---------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Credit Score", credit_score)

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    with col3:
        if loan_status == "Approved":
            risk = "🟢 Low"
        else:
            risk = "🔴 High"

        st.metric("Risk Level", risk)

    st.progress(confidence / 100)



    # ---------------------------------
    # Risk Assessment
    # ---------------------------------

    st.markdown("## 📊 Risk Assessment")

    if loan_status == "Approved":

        if confidence >= 95:
            st.success("🟢 Risk Level: Low Risk")

        elif confidence >= 80:
            st.warning("🟡 Risk Level: Medium Risk")

        else:
            st.error("🔴 Risk Level: High Risk")

    else:

        if confidence >= 95:
            st.error("🔴 Risk Level: High Risk")

        elif confidence >= 80:
            st.warning("🟡 Risk Level: Medium Risk")

        else:
            st.success("🟢 Risk Level: Low Risk")




    # ---------------------------------
    # Applicant Summary
    # ---------------------------------

    st.markdown("## 📋 Applicant Summary")

    summary = {
        "Applicant Income": f"₹{applicant_income:,}",
        "Coapplicant Income": f"₹{coapplicant_income:,}",
        "Credit Score": credit_score,
        "Loan Amount": f"₹{loan_amount:,}",
        "Savings": f"₹{savings:,}",
        "Employment": employment_status,
        "Loan Purpose": loan_purpose,
        "Property Area": property_area
    }

    summary_df = pd.DataFrame(
        summary.items(),
        columns=["Field", "Value"]
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )






with st.expander("📖 About this Project"):

    st.markdown("""
### CreditWise – AI Credit Risk Prediction System

This application predicts whether a loan application is likely to be approved or rejected using Machine Learning.

### Features
- AI-powered prediction
- XGBoost Classifier
- Confidence Score
- Fast and interactive interface

### Model Performance
- Accuracy: **92.5%**
- Precision: **84.8%**
- Recall: **91.8%**
- F1 Score: **88.2%**

### Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
""")  







st.divider()

st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit, XGBoost, and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)