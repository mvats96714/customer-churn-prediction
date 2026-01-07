# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using structured business data.

---

## 🚀 Features
- Data cleaning and preprocessing
- One-hot encoding for categorical variables
- Model comparison: Logistic Regression vs XGBoost
- Best model selection using ROC-AUC
- Production-ready Streamlit web application

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## 📂 Project Structure
churn_project/
│
├── data/
│   └── churn.csv
├── train.py
├── app.py
├── model.pkl
├── scaler.pkl
├── features.pkl
└── README.md

---

## ▶️ How to Run Locally

```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib
python train.py
streamlit run app.py
