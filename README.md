# FactoryGuard AI — IoT Predictive Maintenance Engine

> **Infotact Solutions | Data Intelligence Unit | Cohort Zeta | Q4 2025**

## Problem Statement
A large-scale manufacturing facility operates 500 critical robotic arms.  
Unplanned equipment failure costs **$10,000/hour** in downtime.  
FactoryGuard AI predicts failure **24 hours in advance** from streaming IoT sensor data.

---

## Project Architecture
---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data Engineering | Pandas, NumPy | Ingestion, cleaning, feature creation |
| Baseline Model | Scikit-Learn Logistic Regression | Comparison baseline |
| Advanced Model | XGBoost + RandomizedSearchCV | Production model |
| Ensemble Model | Random Forest | Week 2 comparison |
| Imbalance Handling | SMOTE (imbalanced-learn) | Rare failure oversampling |
| Explainability | SHAP | XAI local + global explanations |
| Deployment | Flask REST API | Model-as-a-Service endpoint |
| Serialization | joblib | Model persistence |
| IDE | VS Code | Production-grade modular code |

---

## Setup
```bash
git clone https://github.com/nisamalani/FactoryGuard-AI.git
cd FactoryGuard-AI
pip install -r requirements.txt
brew install libomp  # macOS only
```

## Run Training
```bash
python3 run_training.py
```

## Start API
```bash
python3 app.py
```

## Test API
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [0.5,75.2,1.1,0.48,74.8,1.09,0.51,75.1,1.12,0.49,74.9,1.1,0.5,75.0,1.11,0.47,74.7,1.08,0.5,75.0,1.1]}'
```

---

## Results

| Model | F1-Score | Recall |
|-------|----------|--------|
| Logistic Regression (baseline) | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 |
| **XGBoost (deployed)** | **1.00** | **1.00** |

> F1=1.0 on clean synthetic data. Real industrial data typically yields F1 of 0.85-0.92.

---

## Key Implementation Details

### Week 1 - Data Engineering
- CSV ingestion, linear interpolation for missing values
- Sorted by machine_id + timestamp to prevent data leakage
- Correlation matrix calculated and visualized

### Week 2 - Modeling
- Logistic Regression baseline
- Random Forest with SMOTE
- XGBoost with RandomizedSearchCV
- Metric: F1-Score + Recall (NOT accuracy)

### Week 3 - Explainability (XAI)
- SHAP TreeExplainer for global feature importance
- SHAP Summary Plot saved
- SHAP Force Plot for individual failure explanation

### Week 4 - Deployment
- Flask REST API returning failure probability + SHAP explanation
- API latency < 50ms
- Model saved with joblib
