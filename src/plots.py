import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import THRESHOLDS, TAG_INFO

def tag_label(tag):
    info = TAG_INFO.get(tag, {})
    return f"{tag} - {info.get('label', tag)}"

def tag_unit(tag):
    return TAG_INFO.get(tag, {}).get("unit", "")

def plot_fault_comparison(df_orig, df_fault, anchor_sensor, title="Fault Analysis"):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.5, 1.2, 0.5])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    fault_mask  = df_fault["anomaly_level"] > 0
    fault_start = df_fault.index[fault_mask].min() if fault_mask.any() else df_fault.index[-1]

    # --- Panel 1: Original vs Faulty ---
    ax1.plot(df_orig.index,  df_orig[anchor_sensor],
             color="gray", alpha=0.5, linestyle="--", label="Original (Baseline)")
    ax1.plot(df_fault.index, df_fault[anchor_sensor],
             color="#d62728", linewidth=1.5, label=f"{anchor_sensor} (With Fault)")

    if anchor_sensor in THRESHOLDS:
        ax1.axhline(THRESHOLDS[anchor_sensor]["warning"],
                    color="orange", linestyle=":", label="Warning Threshold")
        ax1.axhline(THRESHOLDS[anchor_sensor]["trip"],
                    color="red", linestyle=":", label="Trip Threshold")

    ax1.axvspan(fault_start, df_fault.index.max(), color="yellow", alpha=0.08)
    ax1.set_ylabel(f"Value [{tag_unit(anchor_sensor)}]")
    ax1.set_title(f"Impact on Main Sensor: {tag_label(anchor_sensor)}",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # --- Panel 2: Delta + rolling mean ---
    delta  = df_fault[anchor_sensor] - df_orig[anchor_sensor]
    window = max(10, len(delta) // 60)
    ax2.fill_between(df_fault.index, 0, delta, color="red", alpha=0.2, label="Raw delta")
    ax2.plot(df_fault.index, delta, color="red", linewidth=0.8, alpha=0.5)
    ax2.plot(df_fault.index, delta.rolling(window, center=True).mean(),
             color="#8B0000", linewidth=1.8, label=f"Rolling mean (w={window})")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvspan(fault_start, df_fault.index.max(), color="yellow", alpha=0.08)
    ax2.set_ylabel("Delta (Fault Intensity)")
    ax2.set_title("Stochastic Ramp Component (The 'Delta' added to Signal)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)

    # --- Panel 3: Anomaly level bar ---
    LEVEL_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
    for lvl, clr in LEVEL_COLORS.items():
        mask = df_fault["anomaly_level"] == lvl
        if mask.any():
            ax3.fill_between(df_fault.index, 0, 1,
                             where=mask, color=clr, step="mid", alpha=0.85)

    ax3.set_yticks([])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Timestamp")
    ax3.set_title("Anomaly State (0: Normal | 1: Incipient | 2: Critical)", fontsize=10)

    fig.suptitle(f"Detailed Fault Injection Report: {title}",
                 fontsize=16, fontweight="bold")
    plt.show()
