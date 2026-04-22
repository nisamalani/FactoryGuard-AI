# 🚀 FactoryGuard AI

<<<<<<< HEAD
> **Infotact Solutions | Data Intelligence Unit | Cohort Zeta | Q4 2026**

## Problem Statement
A large-scale manufacturing facility operates 500 critical robotic arms.  
Unplanned equipment failure costs **$10,000/hour** in downtime.  
FactoryGuard AI predicts failure **24 hours in advance** from streaming IoT sensor data.
=======
Production-ready Predictive Maintenance System that predicts machine failure using sensor data and explains predictions using SHAP.
>>>>>>> 00b67ff (Final project update)

---

## 📌 Features

- Predict machine failure (binary classification)
- Handle imbalanced data using SMOTE
- Explain predictions using SHAP
- Flask API for real-time prediction
- Feature engineering (rolling + lag features)

---

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Flask

---

## 📂 Project Structure

FactoryGuard_AI/
│
├── data/
│   └── sensor_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── explain.py
│
├── models/
│   └── model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## ▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train.py

Run API:

python app.py

---

## 🌐 API

POST /predict

### Input

{
  "temperature": 85,
  "vibration": 0.8,
  "pressure": 30,
  "temp_roll_mean": 82,
  "vibration_roll_mean": 0.75,
  "temp_lag1": 84,
  "vibration_lag1": 0.7
}

### Output

{
  "failure_prediction": 1,
  "failure_probability": 1.0,
  "top_factors": ["pressure", "temperature"]
}

---

## 📊 Explainability

Uses SHAP to explain model predictions.

---

## 👨‍💻 Author

Nisha Malani

---

Production Machine Learning Project