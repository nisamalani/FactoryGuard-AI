import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # ✅ handle missing values properly
    df = df.ffill()

    # optional: drop remaining NaN
    df = df.dropna()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df