import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

"""Binary Logistic Regression"""

def train_logistic_regression(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "is_injected_anomaly",
) -> Pipeline:
    """
    Train a binary Logistic Regression model.
    """

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    return model


def apply_logistic_regression(
    df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Apply trained Logistic Regression model.
    """

    df_result = df.copy()

    X = df_result[feature_cols]

    anomaly_score = model.predict_proba(X)[:, 1]
    pred_anomaly = (anomaly_score >= threshold).astype(int)

    df_result["anomaly_score"] = anomaly_score
    df_result["pred_anomaly"] = pred_anomaly

    return df_result