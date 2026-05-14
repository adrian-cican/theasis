import pandas as pd
import numpy as np
from utils import THRESHOLDS


def _delta_ramp(total_l: int, delta: float, sensor_std: float,
                smooth_window: int = 10) -> np.ndarray:
    trend       = np.linspace(0, delta, total_l)
    noise_scale = np.linspace(sensor_std * 0.1, sensor_std * 0.5, total_l)
    noise       = np.random.normal(0, noise_scale, total_l)
    ramp        = trend + noise
    # Smooth to avoid unrealistic high-frequency spikes on the trend
    return pd.Series(ramp).rolling(smooth_window, center=True, min_periods=1).mean().values


def _delta_oscillation(total_l: int, delta: float, sensor_std: float,
                        period_samples: int = 30) -> np.ndarray:
    """
    Sinusoidal oscillation with a growing amplitude envelope.
    Simulates progressive valve instability — starts subtle and
    becomes more severe over time. Noise is fixed at 0.5x sensor std.
    """
    t        = np.arange(total_l)
    envelope = np.linspace(0.1, 1.0, total_l)
    noise    = np.random.normal(0, sensor_std * 0.5, total_l)
    return delta * np.sin(2 * np.pi * t / period_samples) * envelope + noise


def _delta_spike(total_l: int, delta: float, sensor_std: float,
                 center_frac: float = 0.5, width_frac: float = 0.08) -> np.ndarray:
    """
    Localised Gaussian spike centred at center_frac of the fault window.
    Models a fast isolated thermal event (e.g. defective bearing).
    Width controls sharpness — smaller width_frac = more abrupt spike.
    Noise is fixed at 0.3x sensor std.
    """
    center = int(total_l * center_frac)
    width  = max(int(total_l * width_frac), 1)
    t      = np.arange(total_l)
    noise  = np.random.normal(0, sensor_std * 0.3, total_l)
    return delta * np.exp(-0.5 * ((t - center) / width) ** 2) + noise


_DELTA_FN = {
    "ramp":        _delta_ramp,
    "oscillation": _delta_oscillation,
    "spike":       _delta_spike,
}


def fault_injection(
    df:             pd.DataFrame,
    main_sensor:    str,
    length_time:    float,
    start_from:     float,
    fault_shape:    str   = "ramp",
    corr_threshold: float = 0.5,
    decay_rate:     float = 0.3,
    **shape_kwargs,
) -> pd.DataFrame:
    """
    Injects a synthetic fault into main_sensor and propagates it to
    correlated sensors via Pearson correlation.

    Parameters
    ----------
    df             : original session DataFrame
    main_sensor    : primary sensor to perturb
    length_time    : duration of the fault ramp phase in hours
    start_from     : fault start position as a fraction of the session [0, 1]
    fault_shape    : shape of the injected delta — "ramp" | "oscillation" | "spike"
    corr_threshold : minimum |r| for a sensor to receive propagated error
    decay_rate     : exponential decay constant after the ramp phase ends;
                     higher value = faster return toward baseline
    """
    warning_level = THRESHOLDS[main_sensor]["warning"]
    trip_level    = THRESHOLDS[main_sensor]["trip"]

    data      = df.copy()
    n         = len(data)
    start_idx = int(round(n * start_from))
    end_idx   = min(start_idx + int(round(length_time * 60)), n)
    total_l   = end_idx - start_idx

    corr         = data.corr(method="pearson", numeric_only=True)[main_sensor]
    corr_sensors = corr[abs(corr) >= corr_threshold].index.tolist()

    avg        = data[main_sensor].iloc[start_idx: start_idx + 10].mean()
    delta      = trip_level - avg
    sensor_std = data[main_sensor].iloc[:start_idx].std()
    ramp       = _DELTA_FN[fault_shape](total_l, delta, sensor_std, **shape_kwargs)

    for col in corr_sensors:
        if col in data.columns:
            data.iloc[start_idx:end_idx, data.columns.get_loc(col)] += ramp * corr[col]

    if end_idx < n:
        remaining = n - end_idx
        decay     = ramp[-1] * np.exp(-decay_rate * np.linspace(0, 1, remaining))
        for col in corr_sensors:
            if col in data.columns:
                data.iloc[end_idx:, data.columns.get_loc(col)] += decay * corr[col]

    data["anomaly_level"] = 0
    injected = data[main_sensor].iloc[start_idx:]
    labels   = np.where(injected >= trip_level, 2,
               np.where(injected >= warning_level, 1, 0))
    data.iloc[start_idx:, data.columns.get_loc("anomaly_level")] = labels

    return data


def f1_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F1 — Clogged Exhaust Filters: slow pressure ramp on PT-903."""
    return fault_injection(df, "PT-903", length_time, start_from, fault_shape="ramp")


def f2_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F2 — Insufficient Cooling: slow thermal ramp on TT-901."""
    return fault_injection(df, "TT-901", length_time, start_from, fault_shape="ramp")


def f3_fault(df: pd.DataFrame, length_time: float, start_from: float,
             period_samples: int = 30) -> pd.DataFrame:
    """F3 — Float Valve Fault: oscillating pressure on PT-903."""
    return fault_injection(df, "PT-903", length_time, start_from,
                           fault_shape="oscillation", period_samples=period_samples)


def f4_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F4 — Clogged Inlet Filter: PT-901 has no thresholds defined — TODO: review with Hélder."""
    raise NotImplementedError("F4 requires threshold definitions for PT-901. Pending review.")


def f5_fault(df: pd.DataFrame, length_time: float, start_from: float,
             center_frac: float = 0.5) -> pd.DataFrame:
    """F5 — Defective Bearing: localised Gaussian spike on TT-904."""
    return fault_injection(df, "TT-904", length_time, start_from,
                           fault_shape="spike", center_frac=center_frac)


def combine_faults(df_orig: pd.DataFrame, fault_fns: list) -> pd.DataFrame:
    """
    Applies multiple fault injections sequentially and merges anomaly labels
    by taking the maximum level across all faults (most severe wins).

    Parameters
    ----------
    df_orig   : original clean session
    fault_fns : list of (fault_fn, kwargs) tuples
    """
    # Start from original, accumulate deltas
    df_combined = df_orig.copy()
    df_combined["anomaly_level"] = 0

    for fault_fn, kwargs in fault_fns:
        df_fault = fault_fn(df_orig, **kwargs)  # always inject on original

        # Add the sensor deltas
        sensor_cols = [c for c in df_orig.select_dtypes(include=np.number).columns
                       if c != "anomaly_level"]
        for col in sensor_cols:
            delta = df_fault[col] - df_orig[col]
            df_combined[col] += delta

        # Merge anomaly labels — most severe wins
        df_combined["anomaly_level"] = np.maximum(
            df_combined["anomaly_level"],
            df_fault["anomaly_level"]
        )

    return df_combined