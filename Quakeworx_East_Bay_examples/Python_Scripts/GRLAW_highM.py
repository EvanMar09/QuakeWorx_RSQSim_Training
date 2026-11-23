#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:56:39 2026

@author: emarschall
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RSQSim Gutenberg-Richter Law Fitting

======================================
Dual methods (MLE + Ordinary LS) with improved magnitude of completeness
detection, fit quality assessment, and Mmax reporting in legends.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Set global font properties to Arial, size 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 10

# --- Configuration ---
indirs = [
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Connected_run_uniform3_low_fa/",
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Disconnected_run_uniform3_low_fa//"
]

outdir = "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/figures_SRL/GR_highM/"

filenames = [
    "connected_uni",
    "disconnected_uni",
]

Model_Type = "Uniform low_fa"
Model_names = ["Connected", "Disconnected"]

os.makedirs(outdir, exist_ok=True)
print(f"Output directory: {outdir}\n")

# --- Formatting parameters ---
font_size = 6
colors = ["blue", "red"]
markers = ["o", "s"]
beginning_string = "%%% end input files"

# Annotation offsets (x, y) in points: [Blue (Connected), Red (Disconnected)]
annot_offsets = [(8, 8), (-12, 15)]

# Initialize figures for MLE and LS methods
fig_mle, ax_mle = plt.subplots(figsize=(5, 2))
fig_ls, ax_ls = plt.subplots(figsize=(5, 2))

# Storage for summary statistics
summary_stats = []


def estimate_mc_inflection(bin_edges, cumulative_rates):
    """Alternative Mc estimation using inflection point in log-log space.

    Returns magnitude where curvature is steepest (rollover point).
    """
    if len(cumulative_rates) < 3:
        return bin_edges[0]

    log_rates = np.log10(np.maximum(cumulative_rates, 1e-6))

    if len(log_rates) > 2:
        d2_rates = np.diff(log_rates, n=2)
        if len(d2_rates) > 0:
            curvature_idx = np.argmax(np.abs(d2_rates))
            return bin_edges[curvature_idx + 1]

    return bin_edges[0]


def compute_residuals_and_r2(observed_log_rates, predicted_log_rates):
    """Compute residuals, R², and RMSE for fit quality assessment."""
    residuals = observed_log_rates - predicted_log_rates
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed_log_rates - np.mean(observed_log_rates)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))

    return residuals, r_squared, rmse


# ==========================================
# MAIN LOOP: PROCESS BOTH MODELS
# ==========================================

for i, model_name in enumerate(Model_names):
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()}")
    print(f"{'='*70}")

    indir = indirs[i]
    filepath = indir + "eqs." + filenames[i] + ".out"

    # --- Locate Data Start ---
    starting_index = []
    with open(filepath, "r") as file:
        for line_num, line in enumerate(file, 0):
            if beginning_string in line:
                starting_index.append(line_num)
                break

    # --- Load and Process Catalog ---
    col_names = [
        "t",
        "Mo",
        "Mw",
        "nuc_x",
        "nuc_y",
        "nuc_z",
        "area",
        "duration",
        "nuc_element",
    ]
    df_events = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        skiprows=starting_index[0] + 1,
        names=col_names,
    )
    df_events.t = df_events.t / (60 * 60 * 24 * 365.25)  # Convert seconds to years

    time_skip = 1 if "Creeping" in Model_Type else 5000
    df = df_events[df_events.t >= time_skip]
    total_years = df.t.max() - df.t.min()

    magnitudes = df["Mw"].values
    magnitudes = magnitudes[np.isfinite(magnitudes)]

    print("\nCatalog Properties:")
    print(f"  Total events: {len(magnitudes)}")
    print(f"  Duration: {total_years:.2f} years")
    print(f"  Magnitude range: {magnitudes.min():.2f} to {magnitudes.max():.2f}")
    print(f"  Mean magnitude: {magnitudes.mean():.2f}")

    # --- Setup Magnitude Bins ---
    bin_width = 0.025
    bins = np.arange(
        np.floor(magnitudes.min()),
        np.ceil(magnitudes.max()) + bin_width,
        bin_width,
    )
    bin_edges = bins[:-1]

    counts, _ = np.histogram(magnitudes, bins=bins)
    cumulative_counts = np.cumsum(counts[::-1])[::-1]
    cumulative_rates = cumulative_counts / total_years

    # --- Dynamic Magnitude of Completeness (Mc) ---
    max_index = np.argmax(counts)
    Mc_maxcurv = bin_edges[max_index]
    Mc_inflection = estimate_mc_inflection(bin_edges, cumulative_rates)

    Mc = Mc_maxcurv
    Mmax = magnitudes.max()
    M_min = 5

    n_below_mc = np.sum(magnitudes < Mc)
    n_above_mc = np.sum(magnitudes >= Mc)
    pct_below_mc = 100 * n_below_mc / len(magnitudes)
    pct_above_mc = 100 * n_above_mc / len(magnitudes)

    print("\nMagnitude of Completeness (Mc):")
    print(f"  Mc (max curvature method): {Mc_maxcurv:.2f}")
    print(f"  Mc (inflection point alt): {Mc_inflection:.2f}")
    print(f"  Using Mc = {Mc:.2f}")
    print(f"  Events with Mw >= Mc: {n_above_mc} ({pct_above_mc:.1f}%)")
    print(f"  Events with Mw < Mc: {n_below_mc} ({pct_below_mc:.1f}%)")
    print(f"  Observed Mmax: {Mmax:.2f}")

    # Set fit line evaluations from M_min (3.5) through Mmax
    fit_line_mags = np.linspace(M_min, Mmax, 100)

    # ==========================================
    # 1. MAXIMUM LIKELIHOOD ESTIMATION (MLE)
    # ==========================================
    fit_mags_mle = magnitudes[magnitudes >= M_min]
    n_mle = len(fit_mags_mle)
    mean_M = np.mean(fit_mags_mle)

    b_value_mle = np.log10(np.e) / (mean_M - M_min)
    sigma_b_mle = b_value_mle / np.sqrt(n_mle)

    rate_M_min = n_mle / total_years
    a_value_mle = np.log10(rate_M_min) + b_value_mle * M_min

    fit_line_log_rate_mle = a_value_mle - b_value_mle * fit_line_mags
    fit_line_rate_mle = 10.0**fit_line_log_rate_mle

    fit_mask_eval = (bin_edges >= M_min) & (cumulative_rates > 0)
    eval_mags = bin_edges[fit_mask_eval]
    eval_obs_log_rates = np.log10(cumulative_rates[fit_mask_eval])
    eval_pred_log_rates = a_value_mle - b_value_mle * eval_mags

    residuals_mle, r2_mle, rmse_mle = compute_residuals_and_r2(
        eval_obs_log_rates, eval_pred_log_rates
    )

    print("\n[MLE] Maximum Likelihood Estimation:")
    print(f"  Events used: {n_mle}")
    print(f"  b-value: {b_value_mle:.4f} ± {sigma_b_mle:.4f}")
    print(f"  a-value: {a_value_mle:.4f}")
    print(f"  Annual rate at M_min ({M_min}): {rate_M_min:.4e} yr⁻¹")
    print("  Fit quality:")
    print(f"    R²: {r2_mle:.4f}")
    print(f"    RMSE (log₁₀): {rmse_mle:.4f}")

    # ==========================================
    # 2. LEAST SQUARES (LS) - UNWEIGHTED
    # ==========================================
    fit_mask_ls = (bin_edges >= M_min) & (cumulative_rates > 0)
    fit_bin_mags_ls = bin_edges[fit_mask_ls]
    fit_log_rates_ls = np.log10(cumulative_rates[fit_mask_ls])

    # Ordinary (Unweighted) Least Squares Fit
    coeffs_ls = np.polyfit(fit_bin_mags_ls, fit_log_rates_ls, 1)
    b_slope_ls, a_intercept_ls = coeffs_ls
    b_value_ls = -b_slope_ls
    a_value_ls = a_intercept_ls

    fit_line_log_rate_ls = a_value_ls - b_value_ls * fit_line_mags
    fit_line_rate_ls = 10.0**fit_line_log_rate_ls

    poly_ls = np.poly1d(coeffs_ls)
    pred_log_rates_ls = poly_ls(fit_bin_mags_ls)
    residuals_ls, r2_ls, rmse_ls = compute_residuals_and_r2(
        fit_log_rates_ls, pred_log_rates_ls
    )

    print("\n[LS] Ordinary Least Squares (Unweighted):")
    print(f"  Bins used: {np.sum(fit_mask_ls)}")
    print(f"  b-value: {b_value_ls:.4f}")
    print(f"  a-value: {a_value_ls:.4f}")
    print("  Fit quality:")
    print(f"    R²: {r2_ls:.4f}")
    print(f"    RMSE (log₁₀): {rmse_ls:.4f}")

    # Store for export
    summary_stats.append({
        "Model": model_name,
        "Mc": Mc,
        "Mmax": Mmax,
        "n_events_total": len(magnitudes),
        "n_events_Mc+": n_above_mc,
        "pct_above_Mc": pct_above_mc,
        "catalog_years": total_years,
        "b_mle": b_value_mle,
        "sigma_b_mle": sigma_b_mle,
        "a_mle": a_value_mle,
        "r2_mle": r2_mle,
        "rmse_mle": rmse_mle,
        "b_ls": b_value_ls,
        "a_ls": a_value_ls,
        "r2_ls": r2_ls,
        "rmse_ls": rmse_ls,
    })

    # Annotation position and offset selection
    text_idx = int(len(fit_line_mags) * (0.65 - i * 0.25))
    xy_offset = annot_offsets[i]

    # ==========================================
    # PLOTTING - MLE
    # ==========================================
    ax_mle.scatter(
        bin_edges,
        cumulative_rates,
        color=colors[i],
        marker=markers[i],
        s=3,
        facecolors="none",
        alpha=0.25,
        edgecolors=colors[i],
        linewidths=0.4,
        label=f"{model_name} Data",
        zorder=3,
    )
    ax_mle.plot(
        fit_line_mags,
        fit_line_rate_mle,
        color=colors[i],
        linestyle="-",
        linewidth=1.5,
        label=f"{model_name} Fit ($M_{{max}} = {Mmax:.2f}$)",
        zorder=4,
    )

    annot_text = f"b = {b_value_mle:.3f}\nR² = {r2_mle:.3f}"
    ax_mle.annotate(
        annot_text,
        xy=(fit_line_mags[text_idx], fit_line_rate_mle[text_idx]),
        xytext=xy_offset,
        textcoords="offset points",
        fontsize=font_size,
        color=colors[i],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.75,
            edgecolor=colors[i],
            linewidth=0.5,
        ),
    )

    # ==========================================
    # PLOTTING - LS
    # ==========================================
    ax_ls.scatter(
        bin_edges,
        cumulative_rates,
        color=colors[i],
        marker=markers[i],
        s=3,
        facecolors="none",
        alpha=0.25,
        edgecolors=colors[i],
        linewidths=0.4,
        label=f"{model_name} Data",
        zorder=3,
    )
    ax_ls.plot(
        fit_line_mags,
        fit_line_rate_ls,
        color=colors[i],
        linestyle="-",
        linewidth=1.5,
        label=f"{model_name} Fit ($M_{{max}} = {Mmax:.2f}$)",
        zorder=4,
    )

    annot_text = f"b = {b_value_ls:.3f}\nR² = {r2_ls:.3f}"
    ax_ls.annotate(
        annot_text,
        xy=(fit_line_mags[text_idx], fit_line_rate_ls[text_idx]),
        xytext=xy_offset,
        textcoords="offset points",
        fontsize=font_size,
        color=colors[i],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.75,
            edgecolor=colors[i],
            linewidth=0.5,
        ),
    )


# ==========================================
# Figure Formatting & Saving
# ==========================================

# MLE Figure Formatting
ax_mle.set_yscale("log")
ax_mle.set_xlim(left=5,right = 8)
ax_mle.xaxis.set_major_locator(MultipleLocator(0.5))
ax_mle.set_xlabel("Magnitude $M_w$", fontsize=font_size, fontweight="bold")
ax_mle.set_ylabel(
    "Cumulative Annual Rate (yr$^{-1}$)", fontsize=font_size, fontweight="bold"
)
ax_mle.set_title(
    f"Magnitude-Frequency Distribution - Model: {Model_Type}",
    fontweight="bold",
    fontsize=font_size,
    pad=10,
)
ax_mle.tick_params(axis="both", labelsize=font_size)
ax_mle.grid(True, which="major", linestyle="-", alpha=0.2, linewidth=0.5)
ax_mle.grid(True, which="minor", linestyle=":", alpha=0.1, linewidth=0.3)
ax_mle.legend(
    fontsize=font_size,
    loc="lower left",
    framealpha=0.95,
    edgecolor="black",
    fancybox=False,
)
fig_mle.tight_layout()

mle_outpath = f"{outdir}01_MLE_RSQSim_Gutenberg_Richter_{Model_Type.replace(' ', '_')}_Comparison.png"
fig_mle.savefig(mle_outpath, dpi=300, bbox_inches="tight")
print(f"\n✓ Saved MLE plot: {mle_outpath}")

# LS Figure Formatting
ax_ls.set_yscale("log")
ax_ls.set_xlim(left=3.5)
ax_ls.xaxis.set_major_locator(MultipleLocator(0.5))
ax_ls.set_xlabel("Magnitude $M_w$", fontsize=font_size, fontweight="bold")
ax_ls.set_ylabel(
    "Cumulative Annual Rate (yr$^{-1}$)", fontsize=font_size, fontweight="bold"
)
ax_ls.set_title(
    f"Gutenberg-Richter MFD: Least Squares (Unweighted)\n{Model_Type}",
    fontweight="bold",
    fontsize=font_size,
    pad=10,
)
ax_ls.tick_params(axis="both", labelsize=font_size)
ax_ls.grid(True, which="major", linestyle="-", alpha=0.2, linewidth=0.5)
ax_ls.grid(True, which="minor", linestyle=":", alpha=0.1, linewidth=0.3)
ax_ls.legend(
    fontsize=font_size,
    loc="lower left",
    framealpha=0.95,
    edgecolor="black",
    fancybox=False,
)
fig_ls.tight_layout()

ls_outpath = f"{outdir}02_LS_RSQSim_Gutenberg_Richter_{Model_Type.replace(' ', '_')}_Comparison.png"
#fig_ls.savefig(ls_outpath, dpi=300, bbox_inches="tight")
print(f"✓ Saved LS plot:  {ls_outpath}")

# Export Summary Statistics
summary_df = pd.DataFrame(summary_stats)
csv_outpath = f"{outdir}GR_Fit_Summary.csv"
#summary_df.to_csv(csv_outpath, index=False)
print(f"✓ Saved summary:  {csv_outpath}\n")

plt.show()