import shap
import joblib
import pandas as pd

def explain():
    model = joblib.load('models/model.pkl')

    df = pd.read_csv('data/sensor_data.csv')
    X = df.drop('failure', axis=1)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    explain()