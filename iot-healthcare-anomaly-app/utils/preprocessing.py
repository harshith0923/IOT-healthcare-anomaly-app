import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    df_clean = df.select_dtypes(include=["float64", "int64"]).dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_clean.values)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    return tensor
