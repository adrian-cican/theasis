import pandas as pd
import numpy as np
from utils import THRESHOLDS

def fault_injection(df: pd.DataFrame, main_sensor, length_time: float, start_from: float) -> pd.DataFrame:
    """
    Fault injection logic.
    Propagates the error to sensors with correlation |r| >= 0.50
    :param df:
    :param main_sensor:
    :param length_time:
    :param start_from:
    :return:
    """
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

    # Create the error trend
    avg     = data[main_sensor].iloc[start_idx: start_idx + 10].mean()
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
    injected_values                                                     = data[main_sensor].iloc[start_idx:end_idx]
    labels                                                              = np.where(injected_values >= warning_level, 2, 1)
    data.iloc[start_idx:end_idx, data.columns.get_loc('anomaly_level')] = labels

    # Keep the data faulty
    if end_idx < n:
        for col in corr_sensors:
            final_value = corr[col] * ramp[-1]
            data.iloc[end_idx:, data.columns.get_loc(col)]  += final_value

    return data


def f1_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Clogged Exhaust Filters:
        - Main sensor affected: PT-903
    """
    return fault_injection(df, "PT-903", length_time, start_from)

def f2_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Insufficient Cooling:
        - Main sensor affected: TT-904
    """
    return fault_injection(df, "TT-904", length_time, start_from)

# TODO: implement F3 (needs review)

def f3_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Float Valve Fault:
        -
    """
    return fault_injection(df, "TT-904", length_time, start_from)


# TODO: Needs Adjustments in the Warning/Trips Signals (Check with Helder)
def f4_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Clogged Inlet Filter:
        - Main sensor affected: PT-901
    """
    return fault_injection(df, "PT-901", length_time, start_from)

def f5_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """
        Simulates Defective Bearings:
        - Main sensor affected: TT-904 (Fast localized Spike)
    """
    return fault_injection(df, "TT-904", length_time, start_from)