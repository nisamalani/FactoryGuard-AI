import pandas as pd
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.sort_values(['machine_id', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method='linear', limit_direction='both')
    df.dropna(inplace=True)
    return df
