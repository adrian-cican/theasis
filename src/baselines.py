from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

"""Univariate Z-Score Model"""

def train_zscore_models(
    train_df: pd.DataFrame,
    sensors: list[str]
) -> dict:
    """
    Train one StandardScaler per sensor.
    """

    scalers = {}
    for sensor in sensors:

        scaler = StandardScaler()
        scaler.fit(train_df[[sensor]])
        scalers[sensor] = scaler

    return scalers


def apply_zscore_detection(
    df: pd.DataFrame,
    scalers: dict,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Apply z-score anomaly detection using trained scalers.
    """

    df_result = df.copy()

    for sensor, scaler in scalers.items():

        z_scores = scaler.transform(df[[sensor]])
        z_scores = np.abs(z_scores.flatten())
        df_result[f"{sensor}_zscore"] = z_scores
        df_result[f"{sensor}_anomaly"] = (
            z_scores > threshold
        ).astype(int)

    return df_result