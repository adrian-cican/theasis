from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from baselines import train_zscore_models, apply_zscore_detection

def evaluate_zscore_experiment(
    train_df,
    test_df,
    sensors,
    threshold=3
):
    # train
    scalers = train_zscore_models(
        train_df,
        sensors
    )
    # inference
    results_df = apply_zscore_detection(
        test_df,
        scalers,
        threshold
    )
    # global anomaly
    anomaly_cols = [
        f"{sensor}_anomaly"
        for sensor in sensors
    ]
    results_df["global_anomaly"] = (
        results_df[anomaly_cols]
        .sum(axis=1) > 0
    ).astype(int)

    # ground truth
    y_true = results_df["is_injected_anomaly"]
    y_pred = results_df["global_anomaly"]

    metrics = {
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }
    return metrics, results_df