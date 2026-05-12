import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import THRESHOLDS, TAG_INFO

def tag_label(tag):
    info = TAG_INFO.get(tag, {})
    return f"{tag} - {info.get('label', tag)}"

def tag_unit(tag):
    return TAG_INFO.get(tag, {}).get("unit", "")

def plot_fault_comparison(df_orig, df_fault, anchor_sensor, title="Fault Analysis"):
    plt.rcParams.update({"font.size": 10, "axes.grid": True})
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 0.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Detect fault start for visualization
    fault_mask = df_fault['anomaly_level'] > 0
    fault_start = df_fault.index[fault_mask].min()

    # Panel 1: Original vs Faulty Signal
    ax1.plot(df_orig.index, df_orig[anchor_sensor], label="Original (Baseline)", color='gray', alpha=0.5, linestyle='--')
    ax1.plot(df_fault.index, df_fault[anchor_sensor], label=f"{anchor_sensor} (Faulty)", color='#d62728', linewidth=1.5)

    if anchor_sensor in THRESHOLDS:
        ax1.axhline(THRESHOLDS[anchor_sensor]["warning"], color='orange', linestyle=':', label="Warning Threshold")
        ax1.axhline(THRESHOLDS[anchor_sensor]["trip"], color='red', linestyle=':', label="Trip Threshold")

    ax1.set_title(f"Main Sensor Impact: {tag_label(anchor_sensor)}", fontsize=14, fontweight='bold')
    ax1.set_ylabel(f"Value [{tag_unit(anchor_sensor)}]")
    ax1.legend(loc="upper left")

    # Panel 2: Isolated Injected Error (Delta)
    error_delta = df_fault[anchor_sensor] - df_orig[anchor_sensor]
    ax2.fill_between(df_fault.index, 0, error_delta, color='red', alpha=0.2, label="Injected Delta")
    ax2.plot(df_fault.index, error_delta, color='red', linewidth=1)
    ax2.set_ylabel("Delta Intensity")
    ax2.set_title("Stochastic Ramp Component")

    # Panel 3: Anomaly Levels (0, 1, 2)
    colors = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728'}
    for level in [0, 1, 2]:
        mask = df_fault['anomaly_level'] == level
        if mask.any():
            ax3.scatter(df_fault.index[mask], [1] * mask.sum(), c=colors[level], marker='|', s=500)

    ax3.set_yticks([])
    ax3.set_title("Anomaly State (0: Normal | 1: Incipient | 2: Critical)")
    ax3.set_ylim(0.8, 1.2)

    # Highlight active fault zone across all panels
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(fault_start, df_fault.index.max(), color='yellow', alpha=0.1)

    fig.suptitle(f"Fault Injection Report: {title}", fontsize=18)
    plt.show()