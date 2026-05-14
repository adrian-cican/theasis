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

def _delta_micro_ramp(total_l: int, amplitude: float, sensor_std: float,
                      smooth_window: int = 20) -> np.ndarray:
    """
    Smooth low-amplitude drift with gently increasing intensity.
    This is intended to look like process drift, not a clear fault ramp.
    """
    t = np.linspace(0, 1, total_l)
    smoothstep = 3 * t**2 - 2 * t**3
    noise = np.random.normal(0, sensor_std * 0.08, total_l)
    noise = pd.Series(noise).rolling(smooth_window, center=True, min_periods=1).mean().values
    return amplitude * smoothstep + noise

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

def _delta_micro_oscillation(total_l: int, amplitude: float, sensor_std: float,
                             period_samples: int = 90,
                             smooth_window: int = 8) -> np.ndarray:
    """
    Low-amplitude irregular oscillation around the baseline.
    The slow amplitude modulation avoids a perfectly synthetic sine wave.
    """
    t = np.arange(total_l)
    phase_jitter = pd.Series(
        np.random.normal(0, 0.08, total_l)
    ).rolling(smooth_window, center=True, min_periods=1).mean().values
    envelope = 0.6 + 0.25 * np.sin(2 * np.pi * t / max(period_samples * 4, 1))
    noise = np.random.normal(0, sensor_std * 0.05, total_l)
    return amplitude * envelope * np.sin(2 * np.pi * t / period_samples + phase_jitter) + noise


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

def _delta_micro_bumps(total_l: int, amplitude: float, sensor_std: float,
                       num_bumps: int = 4,
                       width_frac: float = 0.035) -> np.ndarray:
    """
    Several small local bumps, useful for subtle transient anomalies.
    Each bump is below the scale of a hard spike.
    """
    t = np.arange(total_l)
    signal = np.zeros(total_l)
    width = max(int(total_l * width_frac), 1)

    if total_l <= 2:
        return signal

    centers = np.linspace(0.2, 0.8, num_bumps) * (total_l - 1)
    centers += np.random.normal(0, total_l * 0.03, num_bumps)
    for center in centers:
        bump_amplitude = amplitude * np.random.uniform(0.35, 0.8)
        signal += bump_amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

    noise = np.random.normal(0, sensor_std * 0.05, total_l)
    return signal + noise

_DELTA_FN = {
    "ramp":        _delta_ramp,
    "oscillation": _delta_oscillation,
    "spike":       _delta_spike,
}

_SMALL_DELTA_FN = {
    "micro_ramp":        _delta_micro_ramp,
    "micro_oscillation": _delta_micro_oscillation,
    "micro_bumps":       _delta_micro_bumps,
}

def _fault_window(n: int, length_time: float, start_from: float) -> tuple[int, int, int]:
    """Return start/end indices for one-minute sampled sessions."""
    if not 0 <= start_from <= 1:
        raise ValueError("start_from must be between 0 and 1.")
    if length_time <= 0:
        raise ValueError("length_time must be positive.")

    start_idx = int(round(n * start_from))
    end_idx = min(start_idx + int(round(length_time * 60)), n)
    total_l = end_idx - start_idx
    if total_l <= 0:
        raise ValueError("Fault window is empty. Check length_time and start_from.")
    return start_idx, end_idx, total_l


def _small_amplitude(avg: float, warning_level: float, sensor_std: float,
                     amplitude_std: float, warning_fraction: float) -> float:
    """
    Compute a subtle amplitude that usually stays below the warning threshold.
    The sign follows the direction from the local baseline to the warning level.
    """
    if pd.isna(sensor_std) or sensor_std == 0:
        sensor_std = max(abs(avg) * 0.005, 1e-6)

    direction = np.sign(warning_level - avg)
    if direction == 0:
        direction = 1

    distance_to_warning = abs(warning_level - avg)
    amplitude = min(sensor_std * amplitude_std, distance_to_warning * warning_fraction)
    return direction * amplitude


def fault_injection(
        df: pd.DataFrame,
        main_sensor: str,
        length_time: float,
        start_from: float,
        fault_type: str,  # Added to map anomaly_type (e.g., "clogged_filters")
        fault_shape: str = "ramp",
        corr_threshold: float = 0.5,
        decay_rate: float = 0.3,
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
    fault_type     : descriptive name of the fault class for machine learning categorization
    fault_shape    : shape of the injected delta — "ramp" | "oscillation" | "spike"
    corr_threshold : minimum |r| for a sensor to receive propagated error
    decay_rate     : exponential decay constant after the ramp phase ends;
                     higher value = faster return toward baseline
    """
    warning_level = THRESHOLDS[main_sensor]["warning"]
    trip_level = THRESHOLDS[main_sensor]["trip"]

    data = df.copy()
    n = len(data)
    start_idx, end_idx, total_l = _fault_window(n, length_time, start_from)

    # Calculate correlation matrix against anchor
    corr = data.corr(method="pearson", numeric_only=True)[main_sensor]
    corr_sensors = corr[abs(corr) >= corr_threshold].index.tolist()

    # Calculate injection delta based on baseline average
    avg = data[main_sensor].iloc[start_idx: start_idx + 10].mean()
    delta = trip_level - avg
    sensor_std = data[main_sensor].iloc[:start_idx].std()
    ramp = _DELTA_FN[fault_shape](total_l, delta, sensor_std, **shape_kwargs)

    # 1. Propagate fault during active injection phase
    for col in corr_sensors:
        if col in data.columns:
            data.iloc[start_idx:end_idx, data.columns.get_loc(col)] += ramp * corr[col]

    # 2. Propagate fault decay / stabilization phase
    if end_idx < n:
        remaining = n - end_idx
        decay = ramp[-1] * np.exp(-decay_rate * np.linspace(0, 1, remaining))
        for col in corr_sensors:
            if col in data.columns:
                data.iloc[end_idx:, data.columns.get_loc(col)] += decay * corr[col]


    # Target 1: Anomaly Severity Level (Dynamic based on absolute limits)
    data["anomaly_level"] = 0
    injected = data[main_sensor].iloc[start_idx:]
    labels = np.where(injected >= trip_level, 2,
                      np.where(injected >= warning_level, 1, 0))
    data.iloc[start_idx:, data.columns.get_loc("anomaly_level")] = labels

    # Target 2: Oracle Ground Truth (Binary indicator of any signal degradation)
    data["is_injected_anomaly"] = 0
    data.iloc[start_idx:, data.columns.get_loc("is_injected_anomaly")] = 1

    # Target 3: Fault Classification Type (Categorical variable)
    data["anomaly_type"] = "normal"
    data.iloc[start_idx:, data.columns.get_loc("anomaly_type")] = fault_type

    return data


def small_anomaly_injection(
    df: pd.DataFrame,
    main_sensor: str,
    length_time: float,
    start_from: float,
    anomaly_shape: str = "micro_ramp",
    amplitude_std: float = 0.6,
    warning_fraction: float = 0.25,
    corr_threshold: float = 0.65,
    propagation_scale: float = 0.35,
    decay_rate: float = 1.2,
    anomaly_type: str | None = None,
    **shape_kwargs,
) -> pd.DataFrame:
    """
    Inject a subtle anomaly that is usually below process warning/trip thresholds.

    Labels mark the injected time window, not only threshold crossings. This is
    useful for evaluating anomaly detectors that should catch weak deviations.
    """
    warning_level = THRESHOLDS[main_sensor]["warning"]

    data = df.copy()
    n = len(data)
    start_idx, end_idx, total_l = _fault_window(n, length_time, start_from)

    corr = data.corr(method="pearson", numeric_only=True)[main_sensor]
    corr_sensors = corr[abs(corr) >= corr_threshold].index.tolist()

    avg = data[main_sensor].iloc[start_idx:start_idx + 10].mean()
    sensor_std = data[main_sensor].iloc[:start_idx].std()
    amplitude = _small_amplitude(avg, warning_level, sensor_std, amplitude_std, warning_fraction)
    delta = _SMALL_DELTA_FN[anomaly_shape](total_l, amplitude, sensor_std, **shape_kwargs)

    for col in corr_sensors:
        if col in data.columns:
            scale = 1.0 if col == main_sensor else propagation_scale
            data.iloc[start_idx:end_idx, data.columns.get_loc(col)] += delta * corr[col] * scale

    if end_idx < n:
        remaining = n - end_idx
        decay = delta[-1] * np.exp(-decay_rate * np.linspace(0, 1, remaining))
        for col in corr_sensors:
            if col in data.columns:
                scale = 1.0 if col == main_sensor else propagation_scale
                data.iloc[end_idx:, data.columns.get_loc(col)] += decay * corr[col] * scale

    data["anomaly_level"] = 0
    data["is_injected_anomaly"] = 0
    data["anomaly_type"] = "normal"

    label = anomaly_type or f"{main_sensor}_{anomaly_shape}"
    data.iloc[start_idx:end_idx, data.columns.get_loc("anomaly_level")] = 1
    data.iloc[start_idx:end_idx, data.columns.get_loc("is_injected_anomaly")] = 1
    data.iloc[start_idx:end_idx, data.columns.get_loc("anomaly_type")] = label

    return data

# --- F1: Clogged Exhaust Filters ---
def f1_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    return fault_injection(
        df, "PT-903", length_time, start_from,
        fault_type="clogged_filters",
        fault_shape="ramp"
    )

def f1_small(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    return small_anomaly_injection(
        df, "PT-903", length_time, start_from,
        anomaly_shape="micro_ramp",
        amplitude_std=0.5,
        warning_fraction=0.20,
        corr_threshold=0.70,
        propagation_scale=0.25,
        anomaly_type="clogged_filters",
    )

# --- F2: Insufficient Cooling ---
def f2_fault(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    return fault_injection(
        df, "TT-901", length_time, start_from,
        fault_type="cooling_fault",
        fault_shape="ramp"
    )

def f2_small(df: pd.DataFrame, length_time: float, start_from: float) -> pd.DataFrame:
    return small_anomaly_injection(
        df, "TT-901", length_time, start_from,
        anomaly_shape="micro_ramp",
        amplitude_std=0.6,
        warning_fraction=0.18,
        corr_threshold=0.60,
        propagation_scale=0.30,
        anomaly_type="cooling_fault",
    )

# --- F3: Float Valve Fault ---
def f3_fault(df: pd.DataFrame, length_time: float, start_from: float, period_samples: int = 30) -> pd.DataFrame:
    return fault_injection(
        df, "PT-903", length_time, start_from,
        fault_type="valve_fault",
        fault_shape="oscillation",
        period_samples=period_samples
    )

def f3_small(df: pd.DataFrame, length_time: float, start_from: float, period_samples: int = 90) -> pd.DataFrame:
    return small_anomaly_injection(
        df, "PT-903", length_time, start_from,
        anomaly_shape="micro_oscillation",
        amplitude_std=0.45,
        warning_fraction=0.15,
        corr_threshold=0.70,
        propagation_scale=0.20,
        period_samples=period_samples,
        anomaly_type="valve_fault",
    )

# --- F4: Defective Bearing ---
def f4_fault(df: pd.DataFrame, length_time: float, start_from: float, center_frac: float = 0.5) -> pd.DataFrame:
    return fault_injection(
        df, "TT-904", length_time, start_from,
        fault_type="bearing_fault",
        fault_shape="spike",
        center_frac=center_frac
    )

def f4_small(df: pd.DataFrame, length_time: float, start_from: float, num_bumps: int = 4) -> pd.DataFrame:
    return small_anomaly_injection(
        df, "TT-904", length_time, start_from,
        anomaly_shape="micro_bumps",
        amplitude_std=0.7,
        warning_fraction=0.12,
        corr_threshold=0.65,
        propagation_scale=0.15,
        num_bumps=num_bumps,
        anomaly_type="bearing_fault",
    )


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
    df_combined["is_injected_anomaly"] = 0
    df_combined["anomaly_type"] = "normal"

    for fault_fn, kwargs in fault_fns:
        df_fault = fault_fn(df_orig, **kwargs)  # always inject on original

        # Add the sensor deltas
        sensor_cols = [c for c in df_orig.select_dtypes(include=np.number).columns
                       if c not in ["anomaly_level", "is_injected_anomaly"]]
        for col in sensor_cols:
            delta = df_fault[col] - df_orig[col]
            df_combined[col] += delta

        # Merge anomaly labels — most severe wins
        df_combined["anomaly_level"] = np.maximum(
            df_combined["anomaly_level"],
            df_fault["anomaly_level"]
        )

        if "is_injected_anomaly" in df_fault.columns:
            mask = df_fault["is_injected_anomaly"] == 1
            df_combined.loc[mask, "is_injected_anomaly"] = 1
            if "anomaly_type" in df_fault.columns:
                current = df_combined.loc[mask, "anomaly_type"]
                incoming = df_fault.loc[mask, "anomaly_type"]
                df_combined.loc[mask, "anomaly_type"] = np.where(
                    current.eq("normal"),
                    incoming,
                    current + "+" + incoming
                )

    return df_combined
