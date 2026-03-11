import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

def train_baseline(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_scaled, y_train)
    return lr, scaler

def train_xgboost(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    search = RandomizedSearchCV(xgb, param_grid, n_iter=5, scoring="f1", cv=3, random_state=42, n_jobs=-1)
    search.fit(X_resampled, y_resampled)
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    print(classification_report(y_test, y_pred))
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")

def explain_with_shap(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP explanation done!")
    return explainer, shap_values

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="model.pkl"):
    return joblib.load(path)


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_res, y_res)
    print("Random Forest trained!")
    return rf
