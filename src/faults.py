import pandas as pd
import numpy as np
from utils import THRESHOLDS

def fault_injection(df: pd.DataFrame, main_sensor, length_time: float, start_from: float) -> pd.DataFrame:
    warning_level = THRESHOLDS[main_sensor]["warning"]
    trip_level = THRESHOLDS[main_sensor]["trip"]

    data        = df.copy()
    n           = len(data)
    start_idx   = int(round(n * start_from))        # Starting sample
    length_min  = int(round(length_time * 60))      # Nº of samples we need to change (if we don't run out of samples)
    end_idx     = min(start_idx + length_min, n)    # Ending sample
    total_l     = end_idx - start_idx               # Nº of samples we will actually change

    # Calculate the Pearson Correlation of the main sensor
    corr = data.corr(method="pearson", numeric_only=True)[main_sensor]



def f1_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Clogged Exhaust Filters:
        - Main sensor affected: PT-903
        - Propagates the error to sensors with correlation |r| >= 0.50
    """

    warning_level = THRESHOLDS["PT-903"]["warning"]
    trip_level = THRESHOLDS["PT-903"]["trip"]

    data        = df.copy()
    n           = len(data)
    start_idx   = int(round(n * start_from))        # Starting sample
    length_min  = int(round(length_time * 60))      # Nº of samples we need to change (if we don't run out of samples)
    end_idx     = min(start_idx + length_min, n)    # Ending sample
    total_l     = end_idx - start_idx               # Nº of samples we will actually change


    # Calculate the Pearson Correlation of the main sensor
    corr = data.corr(method="pearson", numeric_only=True)["PT-903"]

    # Create the error trend
    avg     = data["PT-903"].iloc[start_idx: start_idx + 10].mean()
    delta   = trip_level - avg
    trend   = np.linspace(0, delta, total_l)
    noise   = np.random.normal(0, delta * 0.15, total_l)
    ramp    = trend + noise

    # Get the correlated sensors and inject the error
    corr_sensors = corr[abs(corr) >= 0.5].index
    for col in corr_sensors:
        if col in data.columns:
            r = corr[col]
            data.iloc[start_idx:end_idx, data.columns.get_loc(col)] += ramp * r


    # Add a label to keep the anomaly state
    # 0 - No anomaly; 1 - Under de warning level; 2 - Above warning level
    data['anomaly_level']                                               = 0
    injected_values                                                     = data["PT-903"].iloc[start_idx:end_idx]
    labels                                                              = np.where(injected_values >= warning_level, 2, 1)
    data.iloc[start_idx:end_idx, data.columns.get_loc('anomaly_level')] = labels

    # Keep the data faulty
    if end_idx < n:
        for col in corr_sensors:
            final_value = corr[col] * ramp[-1]
            data.iloc[end_idx:, data.columns.get_loc(col)]  += final_value

    return data

def f2_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Insufficient Cooling:
        - Main sensor affected: TT-904
        - Propagates the error to sensors with correlation |r| >= 0.50
    """

    warning_level = THRESHOLDS["TT-904"]["warning"]
    trip_level = THRESHOLDS["TT-904"]["trip"]

    data        = df.copy()
    n           = len(data)
    start_idx   = int(round(n * start_from))        # Starting sample
    length_min  = int(round(length_time * 60))      # Nº of samples we need to change (if we don't run out of samples)
    end_idx     = min(start_idx + length_min, n)    # Ending sample
    total_l     = end_idx - start_idx               # Nº of samples we will actually change