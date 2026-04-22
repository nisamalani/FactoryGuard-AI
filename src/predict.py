import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load('models/model.pkl')
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    prob = model.predict_proba(df)

    return prediction[0], prob[0][1]