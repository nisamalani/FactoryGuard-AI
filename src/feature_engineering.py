import pandas as pd

def add_rolling_features(df):
    df = df.copy()
    sensor_cols = ["vibration", "temperature", "pressure"]
    for col in sensor_cols:
        df[f"{col}_roll_mean_1h"] = df.groupby("machine_id")[col].transform(lambda x: x.rolling(1, min_periods=1).mean())
        df[f"{col}_roll_mean_4h"] = df.groupby("machine_id")[col].transform(lambda x: x.rolling(4, min_periods=1).mean())
        df[f"{col}_roll_mean_8h"] = df.groupby("machine_id")[col].transform(lambda x: x.rolling(8, min_periods=1).mean())
        df[f"{col}_ema"] = df.groupby("machine_id")[col].transform(lambda x: x.ewm(span=4).mean())
        df[f"{col}_lag_1"] = df.groupby("machine_id")[col].transform(lambda x: x.shift(1))
        df[f"{col}_lag_2"] = df.groupby("machine_id")[col].transform(lambda x: x.shift(2))
    df.dropna(inplace=True)
    return df

def get_feature_columns(df):
    exclude = ["timestamp", "machine_id", "failure"]
    return [c for c in df.columns if c not in exclude]
