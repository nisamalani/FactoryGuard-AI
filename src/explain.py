import shap
import joblib
import pandas as pd

from data_preprocessing import load_data, clean_data
from feature_engineering import create_features


def run_shap():
    print("🔹 Loading model...")
    model = joblib.load("models/model.pkl")

    print("🔹 Loading data...")
    df = load_data("data/sensor_data.csv")
    df = clean_data(df)
    df = create_features(df)

    X = df.drop("failure", axis=1)
    X = X.select_dtypes(include=["number"])

    print("🔹 Running SHAP...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Summary plot
    shap.summary_plot(shap_values, X)


if __name__ == "__main__":
    run_shap()