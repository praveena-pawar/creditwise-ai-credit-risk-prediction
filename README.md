# 🏦 CreditWise – AI Credit Risk Prediction System

An end-to-end Machine Learning application that predicts whether a loan application is likely to be **Approved** or **Rejected** using an **XGBoost Classifier**.

The project follows a production-style architecture with a **Streamlit frontend**, **FastAPI backend**, and **XGBoost model** deployed separately.

---

## 🚀 Live Demo

### 🌐 Streamlit App
https://creditwise-loan-prediction.streamlit.app/

### ⚡ FastAPI Documentation
https://creditwise-api-praveen.onrender.com/docs

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications every day. Manually evaluating each application can be slow and inconsistent.

CreditWise uses Machine Learning to analyze applicant information and predict loan eligibility, helping streamline the decision-making process.

---

## ✨ Features

- 🤖 AI-powered Loan Approval Prediction
- 📊 Confidence Score
- ⚠️ Risk Assessment
- 📋 Applicant Summary
- 🎯 Input Validation
- 🌐 REST API using FastAPI
- 💻 Interactive Streamlit Dashboard
- ☁️ Cloud Deployment

---

## 🏗 System Architecture

```text
                 User
                   │
                   ▼
      Streamlit Frontend
                   │
          HTTP POST Request
                   │
                   ▼
      FastAPI REST API (Render)
                   │
                   ▼
        XGBoost Prediction Model
                   │
                   ▼
         Loan Decision + Confidence
```

---

## 📊 Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- Label Encoding
- One-Hot Encoding
- Train-Test Split
- Model Training
- Model Evaluation
- Model Deployment

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **92.5%** |
| Precision | **84.8%** |
| Recall | **91.8%** |
| F1 Score | **88.2%** |

---

## 🛠 Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

### Backend

- FastAPI
- Pydantic

### Frontend

- Streamlit

### Deployment

- Render
- Streamlit Community Cloud

### Tools

- Git
- GitHub

---

## 📂 Project Structure

```text
CreditWise/
│
├── backend/
│   └── app.py
│
├── data/
│   ├── loan_approval_data.csv
│   └── processed_data.csv
│
├── models/
│   └── xgboost_model.pkl
│
├── notebook/
│   └── CreditWise_Model.ipynb
│
├── screenshots/
│   ├── home.png
│   ├── approved.png
│   ├── rejected.png
│   └── api_docs.png
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/praveena-pawar/creditwise-ai-credit-risk-prediction.git
```

Move into the project

```bash
cd creditwise-ai-credit-risk-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run FastAPI

```bash
uvicorn backend.app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 🔗 API Endpoint

### POST /predict

Predict loan approval.

Example Response

```json
{
    "loan_status": "Approved",
    "confidence": "98.72%"
}
```

## 📸 Screenshots

### 🏠 Home Page

![Home Page](screenshots/home-page.png)

---

### ✅ Loan Approved

![Loan Approved](screenshots/approved-loan.png)

---

### ❌ Loan Rejected

![Loan Rejected](screenshots/rejected-loan.png)

---

### 📚 Applicant Summary

![API Docs](screenshots/applicant_summary.png)

## 🎯 Future Improvements

- User Authentication
- Database Integration
- Model Monitoring
- Docker Support
- CI/CD Pipeline
- Cloud Storage Integration

---

## 👨‍💻 Author

**Praveena Pawar**

GitHub:
https://github.com/praveena-pawar



---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
