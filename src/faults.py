import pandas as pd
import numpy as np
from utils import THRESHOLDS


def _delta_ramp(total_l: int, delta: float) -> np.ndarray:
    """
    Linear ramp from 0 → delta with noise.
    Models slow progressive faults (clogged filters, cooling degradation).
    σ grows from 5% → 20% of delta as the fault develops.
    """
    trend = np.linspace(0, delta, total_l)
    noise_scale = np.linspace(abs(delta) * 0.05, abs(delta) * 0.20, total_l)
    noise = np.random.normal(0, noise_scale, total_l)
    return trend + noise


def _delta_oscillation(total_l: int, delta: float, period_samples: int = 30) -> np.ndarray:
    """
    Sinusoidal oscillation with growing amplitude envelope.
    Models float valve instability — the valve degrades progressively,
    so the oscillation amplitude grows over time.

    envelope = linspace(0.1, 1.0) ensures the fault starts subtly
    and becomes more severe.
    """
    t = np.arange(total_l)
    envelope = np.linspace(0.1, 1.0, total_l)
    noise = np.random.normal(0, abs(delta) * 0.05, total_l)
    return delta * np.sin(2 * np.pi * t / period_samples) * envelope + noise


def _delta_spike(total_l: int, delta: float,
                 center_frac: float = 0.5, width_frac: float = 0.08) -> np.ndarray:
    """
    Localised Gaussian spike.
    Models defective bearing — a fast, isolated thermal event.

    Gaussian: delta * exp(-0.5 * ((t - center) / width)²)
    - center : where the peak occurs (fraction of total_l)
    - width  : controls sharpness — smaller = more abrupt spike
    """
    center = int(total_l * center_frac)
    width = max(int(total_l * width_frac), 1)
    t = np.arange(total_l)
    return delta * np.exp(-0.5 * ((t - center) / width) ** 2)


_DELTA_FN = {
    "ramp": _delta_ramp,
    "oscillation": _delta_oscillation,
    "spike": _delta_spike,
}

def fault_injection(
        df: pd.DataFrame,
        main_sensor: str,
        length_time: float,  # hours
        start_from: float,  # fraction of session [0, 1]
        fault_shape: str = "ramp",  # "ramp" | "oscillation" | "spike"
        corr_threshold: float = 0.5,
        decay_rate: float = 0.3,  # exponential decay constant post-ramp
        **shape_kwargs,
) -> pd.DataFrame:
    """
    Injects a synthetic fault into *main_sensor* and propagates it
    to correlated sensors via Pearson correlation.

    Parameters
    ----------
    df             : original session DataFrame
    main_sensor    : primary sensor to perturb
    length_time    : duration of the ramp phase in hours
    start_from     : where in the session the fault starts (0 = beginning, 1 = end)
    fault_shape    : shape of the injected delta signal
    corr_threshold : minimum |r| for a sensor to receive propagated error
    decay_rate     : exponential decay constant for the post-ramp plateau
                     higher = faster decay back toward baseline
    """

    warning_level = THRESHOLDS[main_sensor]["warning"]
    trip_level = THRESHOLDS[main_sensor]["trip"]

    data = df.copy()
    n = len(data)
    start_idx = int(round(n * start_from))
    length_min = int(round(length_time * 60))
    end_idx = min(start_idx + length_min, n)
    total_l = end_idx - start_idx


    # Correlations
    corr = data.corr(method="pearson", numeric_only=True)[main_sensor]
    corr_sensors = corr[abs(corr) >= corr_threshold].index.tolist()

    # Delta signal
    avg = data[main_sensor].iloc[start_idx: start_idx + 10].mean()
    delta = trip_level - avg
    ramp = _DELTA_FN[fault_shape](total_l, delta, **shape_kwargs)

    # Inject ramp phase
    for col in corr_sensors:
        if col in data.columns:
            r = corr[col]
            data.iloc[start_idx:end_idx, data.columns.get_loc(col)] += ramp * r

    # Post-ramp: exponential decay instead of flat plateau
    if end_idx < n:
        remaining = n - end_idx
        t_norm = np.linspace(0, 1, remaining)
        decay = ramp[-1] * np.exp(-decay_rate * t_norm)
        for col in corr_sensors:
            if col in data.columns:
                r = corr[col]
                data.iloc[end_idx:, data.columns.get_loc(col)] += decay * r

    # Anomaly labels
    # Covers both the ramp phase and the decay phase
    data["anomaly_level"] = 0
    injected = data[main_sensor].iloc[start_idx:]
    labels = np.where(injected >= trip_level, 2,
                      np.where(injected >= warning_level, 1, 0))
    data.iloc[start_idx:, data.columns.get_loc("anomaly_level")] = labels

    return data


# Named fault wrappers

def f1_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F1 — Clogged Exhaust Filters: slow pressure ramp in PT-903."""
    return fault_injection(df, "PT-903", length_time, start_from,
                           fault_shape="ramp")


def f2_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F2 — Insufficient Cooling: slow thermal ramp starting at TT-901."""
    return fault_injection(df, "TT-901", length_time, start_from,
                           fault_shape="ramp")


def f3_fault(df: pd.DataFrame, length_time: float, start_from: float,
             period_samples: int = 30) -> pd.DataFrame:
    """F3 — Float Valve Fault: oscillating PT-903 with growing amplitude."""
    return fault_injection(df, "PT-903", length_time, start_from,
                           fault_shape="oscillation",
                           period_samples=period_samples)

# Falar com o Hélder para implementar esta.
def f4_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    """F4 — Clogged Inlet Filter: gradual suction pressure drop in PT-901."""
    return fault_injection(df, "PT-901", length_time, start_from,
                           fault_shape="ramp")


def f5_fault(df: pd.DataFrame, length_time: float, start_from: float,
             center_frac: float = 0.5) -> pd.DataFrame:
    """F5 — Defective Bearing: fast localised spike in TT-904."""
    return fault_injection(df, "TT-904", length_time, start_from,
                           fault_shape="spike",
                           center_frac=center_frac)