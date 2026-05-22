from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models import (
    train_zscore_models,
    apply_zscore_detection,
    train_logistic_regression,
    apply_logistic_regression,
)

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


def evaluate_logistic_regression_experiment(
    train_df,
    test_df,
    feature_cols,
    threshold=0.5
):
    # train
    model = train_logistic_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col="is_injected_anomaly"
    )

    # inference
    results_df = apply_logistic_regression(
        df=test_df,
        model=model,
        feature_cols=feature_cols,
        threshold=threshold
    )

    # ground truth
    y_true = results_df["is_injected_anomaly"]
    y_pred = results_df["pred_anomaly"]
    y_score = results_df["anomaly_score"]

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
        ),
        "num_true_anomalies": int(y_true.sum()),
        "num_predicted_anomalies": int(y_pred.sum()),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics, results_df, model