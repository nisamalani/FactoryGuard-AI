import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report
import joblib
import shap

from data_loader import load_data, clean_data
from feature_engineering import add_rolling_features, get_feature_columns
from model import train_baseline, train_xgboost, train_random_forest, evaluate_model, save_model

def main():
    print("=" * 55)
    print("  FactoryGuard AI - Production Training Pipeline")
    print("  Infotact Solutions | Cohort Zeta | Q4 2026")
    print("=" * 55)

    # Step 1: Load Data
    print("\nStep 1: Loading data...")
    df = load_data("data/sensor_data.csv")
    df = clean_data(df)
    print(f"  Data loaded: {df.shape}")

    # Step 2: Feature Engineering
    print("\nStep 2: Feature engineering...")

    from feature_engineering import apply_feature_engineering

    df = apply_feature_engineering(df)
    feature_cols = get_feature_columns()

    print(f"  Features created: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols[:5]}... (showing first 5)")
    
    # Step 3: Split
    print("\nStep 3: Splitting data (80/20 train/test)...")
    X = df[feature_cols].values
    y = df["failure"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Failure rate in train: {y_train.mean():.2%}")

    # Step 4: Baseline (Logistic Regression)
    print("\nStep 4: Training Logistic Regression baseline...")
    lr_model, scaler = train_baseline(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, scaler=scaler)

    # Step 5: Random Forest
    print("\nStep 5: Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    rf_f1 = f1_score(y_test, rf_model.predict(X_test))
    print(f"  Random Forest F1: {rf_f1:.4f}")

    # Step 6: XGBoost (best model)
    print("\nStep 6: Training XGBoost with RandomizedSearchCV...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)
    xgb_f1 = f1_score(y_test, xgb_model.predict(X_test))
    print(f"  XGBoost F1: {xgb_f1:.4f}")

    # Step 7: Model Comparison
    print("\nStep 7: Model Comparison Summary")
    print(f"  Logistic Regression F1 : {f1_score(y_test, lr_model.predict(scaler.transform(X_test))):.4f}")
    print(f"  Random Forest F1       : {rf_f1:.4f}")
    print(f"  XGBoost F1             : {xgb_f1:.4f}")
    print(f"  Winner: XGBoost (selected for deployment)")

    # Step 8: SHAP Explainability
    print("\nStep 8: SHAP Explainability (XAI)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:200])

    # SHAP Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:200], feature_names=feature_cols, show=False)
    plt.title("SHAP Feature Importance - FactoryGuard AI", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP summary plot saved: shap_summary_plot.png")

    # SHAP Force Plot (bar style for a failure case)
    failure_indices = np.where(y_test == 1)[0]
    if len(failure_indices) > 0:
        idx = failure_indices[0]
        plt.figure(figsize=(12, 4))
        shap_single = shap_values[idx]
        colors = ["#F44336" if v > 0 else "#2196F3" for v in shap_single]
        sorted_idx = np.argsort(np.abs(shap_single))[-10:]
        plt.barh(
            [feature_cols[i] for i in sorted_idx],
            [shap_single[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx]
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"SHAP Force Plot - Sample Failure Case (index {idx})", fontweight="bold")
        plt.xlabel("SHAP Value (impact on failure prediction)")
        plt.tight_layout()
        plt.savefig("shap_force_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP force plot saved: shap_force_plot.png")

    # Step 9: Save Model
    print("\nStep 9: Saving final XGBoost model...")
    save_model(xgb_model, "model.pkl")

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE!")
    print(f"  Final Model: XGBoost | F1: {xgb_f1:.4f} | Recall: {recall_score(y_test, xgb_model.predict(X_test)):.4f}")
    print(f"  Model saved: model.pkl")
    print(f"  Plots: correlation_matrix.png, shap_summary_plot.png, shap_force_plot.png")
    print("=" * 55)

if __name__ == "__main__":
    main()
