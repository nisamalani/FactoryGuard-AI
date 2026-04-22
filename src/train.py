import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from data_preprocessing import load_data, clean_data
from feature_engineering import create_features


def train_model():
    print("🔹 Loading data...")
    df = load_data("data/sensor_data.csv")

    print("🔹 Cleaning data...")
    df = clean_data(df)

    print("🔹 Feature engineering...")
    df = create_features(df)

    # -----------------------------
    # Separate features and target
    # -----------------------------
    X = df.drop("failure", axis=1)

    # Keep only numeric columns (VERY IMPORTANT)
    X = X.select_dtypes(include=["number"])

    y = df["failure"]

    # -----------------------------
    # Train-test split (FIRST!)
    # -----------------------------
    print("🔹 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -----------------------------
    # Apply SMOTE ONLY on training
    # -----------------------------
    print("🔹 Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print(f"After SMOTE → Train shape: {X_train.shape}")

    # -----------------------------
    # Train model
    # -----------------------------
    print("🔹 Training model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Save model
    # -----------------------------
    print("🔹 Saving model...")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model saved at models/model.pkl")


if __name__ == "__main__":
    train_model()