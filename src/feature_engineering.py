def create_features(df):
    # Rolling mean
    df['temp_roll_mean'] = df['temperature'].rolling(window=3).mean()
    df['vibration_roll_mean'] = df['vibration'].rolling(window=3).mean()

    # Lag features
    df['temp_lag1'] = df['temperature'].shift(1)
    df['vibration_lag1'] = df['vibration'].shift(1)

    df = df.dropna()
    return df