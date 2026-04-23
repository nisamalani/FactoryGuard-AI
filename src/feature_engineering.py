def apply_feature_engineering(df):
    # Rolling mean
    df['temp_roll_mean'] = df['temperature'].rolling(window=3).mean()
    df['vibration_roll_mean'] = df['vibration'].rolling(window=3).mean()

    # Lag features
    df['temp_lag1'] = df['temperature'].shift(1)
    df['vibration_lag1'] = df['vibration'].shift(1)

    df = df.dropna()
    return df


def get_feature_columns():
    return [
        'temperature',
        'vibration',
        'temp_roll_mean',
        'vibration_roll_mean',
        'temp_lag1',
        'vibration_lag1'
    ]