#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares RSQSim to UCERF3 by FAULT PARTICIPATION for M >= 6.7 events,
using TIME-INDEPENDENT (TI, Poisson) probabilities only.

This is the same analysis as fault_ucerf3_ti_Mw6_thresh.py with the
time-windowed (TW) calculation and the nucleation comparison removed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- Configuration ---
indirs = [
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Connected_run_uniform3_low_fa/",
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Disconnected_run_uniform3_low_fa/",
]

outdir = "/Users/emarschall/Desktop/RSQSim_stuff/QuakeWorx_test/Stuff_4_Alice_prop/Figs/ucerf__ti_participation_M6thresh/"

filenames = [
    "connected_uni",
    "disconnected_uni",
]
fault_files = [
    "East_Bay_Updated_Connected_HSR3.flt",
    "East_Bay_Updated_Disconnected_HSR3.flt",
]
e_files = [f + ".eList" for f in filenames]
p_files = [f + ".pList" for f in filenames]
d_files = [f + ".dList" for f in filenames]  # per-patch slip, same order/format as eList/pList

Model_Type = "Uniform low_fa"
Model_names = ["Connected", "Disconnected"]

try:
    os.makedirs(outdir, exist_ok=True)
    print(f"Folder '{outdir}' exists.")
except OSError as e:
    print(f"Error creating folder: {e}")

M_threshold = 6.7
time_frame = 30
beginning_string = "%%% end input files"
col_names = ["t", "Mo", "Mw", "nuc_x", "nuc_y", "nuc_z", "area", "duration", "nuc_element"]

# --- Fault participation requires enough rupture on that fault to reach
# a minimum cumulative moment, rather than counting any single patch that
# ruptures. Per-patch moment is Mo_patch = mu * patch_area * slip (dList),
# summed over all patches on that fault within the event; the fault
# "participates" only if that sum reaches the Mw threshold below.
PATCH_SIDE_LENGTH_M = 500.0  # equilateral triangle side length
PATCH_AREA_M2 = (np.sqrt(3) / 4.0) * PATCH_SIDE_LENGTH_M ** 2
SHEAR_MODULUS_PA = 30e9  # mu
FAULT_PARTICIPATION_MW_THRESHOLD = 6.0
# Hanks & Kanamori (SI): Mw = (2/3) * (log10(Mo) - 9.1), Mo in N*m
fault_participation_mo_threshold = 10 ** (1.5 * FAULT_PARTICIPATION_MW_THRESHOLD + 9.1)

results = {}            # per-model TI participation probabilities, by fault
participation_pct = {}
fault_strings = None

# --- Process Both Models ---
for m_idx, model_name in enumerate(Model_names):
    print(f"Processing model: {model_name}...")

    indir = indirs[m_idx]
    infile = "eqs." + filenames[m_idx] + ".out"
    filepath = indir + infile
    fault_filepath = indir + fault_files[m_idx]

    # Locate data start
    starting_index = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 0):
            if beginning_string in line:
                starting_index.append(line_num)
                break

    # Load events
    df_events = pd.read_csv(filepath, sep=" ", header=None, skiprows=starting_index[0] + 1, names=col_names)
    df_events.t = df_events.t / (60 * 60 * 24 * 365.25)

    time_skip = 1 if "Creeping" in Model_Type else 5000
    max_time = int(np.floor(np.max(df_events.t.values)))
    usable_duration = max_time - time_skip

    data_events = np.fromfile(indir + e_files[m_idx], dtype=np.int32)
    data_patches = np.fromfile(indir + p_files[m_idx], dtype=np.int32)
    data_slips = np.fromfile(indir + d_files[m_idx], dtype='<f8')

    # Load fault file
    fault = pd.read_csv(fault_filepath, sep=" ", header=None)
    if fault_strings is None:
        fault_strings = ["Hay", "Cal", "Napa", "Rodgers", "Frank"]
        found = set(np.unique(fault[12]))
        missing = set(fault_strings) - found
        extra = found - set(fault_strings)
        if missing:
            raise ValueError(f"fault_strings expects {missing} but they're not in {fault_filepath}")
        if extra:
            print(f"[WARN] {fault_filepath} has fault names not in fault_strings: {extra} -- they'll be ignored")
    seg_names = fault[12].values

    # Qualifying events
    valid_events_df = df_events[(df_events.Mw >= M_threshold) & (df_events.t >= time_skip) & (df_events.t < max_time)]
    valid_event_ids = valid_events_df.index.values + 1

    participation_event_count = {name: 0 for name in fault_strings}

    for event in valid_event_ids:
        event_indices = np.flatnonzero(np.isin(data_events, event))
        patches = data_patches[event_indices]
        slips = data_slips[event_indices]
        patch_moments = SHEAR_MODULUS_PA * PATCH_AREA_M2 * slips
        patch_fault_names = seg_names[patches - 1]

        # Cumulative moment released on each fault by this event (summed
        # across every patch on that fault that ruptured, not just a count).
        event_fault_moment = pd.Series(patch_moments).groupby(patch_fault_names).sum()

        for name, fault_moment in event_fault_moment.items():
            if name in participation_event_count and fault_moment >= fault_participation_mo_threshold:
                participation_event_count[name] += 1

    total_qualifying_events = len(valid_event_ids)
    participation_pct[model_name] = {
        name: (100 * participation_event_count[name] / total_qualifying_events if total_qualifying_events else 0.0)
        for name in fault_strings
    }

    # ---- PARTICIPATION: time-independent (Poisson) probability ----
    model_time_indep_probs = []
    for name in fault_strings:
        rate = participation_event_count[name] / usable_duration if usable_duration > 0 else 0.0
        model_time_indep_probs.append((1 - np.exp(-rate * time_frame)) * 100)

    results[model_name] = model_time_indep_probs

    print(f"  {total_qualifying_events} qualifying M>={M_threshold} events; "
          f"participation %: " +
          ", ".join(f"{n}={participation_pct[model_name][n]:.1f}%" for n in fault_strings))

#%% --- UCERF3 Probabilities ---
def calc_poisson_probs(rates_list, t):
    """Calculate the joint probability of at least 1 event happening on a fault over time t."""
    probs = []
    for rates in rates_list:
        t_r = 1 - np.exp(-1 * np.array(rates) * t)  # 30 year poisson
        prob_null = 1 - t_r
        prob_null_fault = np.prod(prob_null)
        probs.append((1 - prob_null_fault) * 100)
    return probs


# Raw Annual Participation Rates (M >= 6.7)
# Remember order is Hay, Cal, Napa, Rodgers, Frank
ucerf_rates_mean = [
    [0.0041072527, 0.005948357, 0.0017228004],
    [0.0052234707, 0.0020754202, 0.0041286224],  # 0.004003838],
    [6.631257E-4],
    [0.0032336744],
    [5.71326E-4]
]
ucerf_rates_max = [
    [0.005908122, 0.008005053, 0.004653229],
    [0.01044551, 0.003641902, 0.0072176214],  # , 0.007717815],
    [0.002566178],
    [0.0041210074],
    [0.0038302576]
]
ucerf_rates_min = [
    [0.0025323115, 0.0030774877, 9.0232235e-5],
    [0.0034735922, 0.0015950431, 0.002012016],  # , 0.0021209696],
    [2.275297E-4],
    [0.0022245434],
    [5.254246E-6]
]

# Calculate Probabilities using the Poisson Helper Function
ucerf_probs_mean = calc_poisson_probs(ucerf_rates_mean, time_frame)
ucerf_probs_max = calc_poisson_probs(ucerf_rates_max, time_frame)
ucerf_probs_min = calc_poisson_probs(ucerf_rates_min, time_frame)

# Hayward and Cal correction from UCERF3
ucerf_probs_mean[0] = ucerf_probs_mean[0] - 7
ucerf_probs_max[0] = ucerf_probs_max[0] - 7
ucerf_probs_min[0] = ucerf_probs_min[0] - 7
ucerf_probs_mean[1] = ucerf_probs_mean[1] - 11
ucerf_probs_max[1] = ucerf_probs_max[1] - 11
ucerf_probs_min[1] = ucerf_probs_min[1] - 11

# Create Error arrays for matplotlib
ucerf_mean_arr = np.array(ucerf_probs_mean)
ucerf_yerr = np.vstack([
    ucerf_mean_arr - np.array(ucerf_probs_min),
    np.array(ucerf_probs_max) - ucerf_mean_arr,
])


# Helper function for drawing labels over bars
def add_labels(ax, rects, alignment='center', fs_scaled=6.5):
    for rect in rects:
        height = rect.get_height()
        if pd.isna(height) or height < 0:
            height = 0.0
        if alignment == 'left':
            x_anchor = rect.get_x()
            x_offset = -0.5
        else:
            x_anchor = rect.get_x() + (rect.get_width() / 2)
            x_offset = 0
        ax.annotate(f'{height:.1f}%',
                    xy=(x_anchor, height),
                    xytext=(x_offset, 3), textcoords="offset points",
                    ha=alignment, va='bottom', fontsize=fs_scaled - 1.7, fontweight='semibold', rotation=90)


#%% --- FIGURE: Participation (time-independent only) ---
x = np.arange(len(fault_strings))
width = 0.25
offsets = [-1, 0, 1]
fs_scaled = 6.5

fig, ax = plt.subplots(figsize=(6, 2))

r1 = ax.bar(x + offsets[0] * width, ucerf_probs_mean, width, label='UCERF3',
            color='lightgreen', edgecolor='black', linewidth=0.5)
ax.errorbar(x + offsets[0] * width, ucerf_probs_mean, yerr=ucerf_yerr, fmt='none',
            ecolor='darkgreen', elinewidth=1.2, capsize=2, capthick=1.2)

r2 = ax.bar(x + offsets[1] * width, results['Connected'], width, label='Connected (TI)',
            color='darkblue', edgecolor='black', alpha=0.9, linewidth=0.5)
r3 = ax.bar(x + offsets[2] * width, results['Disconnected'], width, label='Disconnected (TI)',
            color='darkblue', edgecolor='black', hatch='//', alpha=0.9, linewidth=0.5)

ax.set_ylabel(f'{time_frame} Year Probability (%)', fontsize=fs_scaled, fontweight='bold')
ax.set_title(f'{time_frame} Year Fault Participation Probabilities M $\\geq$ {M_threshold} For '
             f'{Model_Type}: Connected vs. Disconnected Catalogs vs. UCERF3',
             fontsize=fs_scaled, fontweight='bold', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(fault_strings, rotation=35, ha='right', fontsize=fs_scaled)
ax.tick_params(axis='both', labelsize=fs_scaled)
ax.set_xlabel('Fault Segment', fontsize=fs_scaled, fontweight='bold', labelpad=0)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
ax.set_ylim(0, 65)  # You may need to adjust this ylim based on your data

# Draw the full box (top and right edges too) and leave a little room on each side
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')
ax.margins(x=0.02)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.65),
    fontsize=fs_scaled - 1.0,
    ncol=3,
    frameon=True,
    framealpha=0.9,
    handletextpad=0.3,
    columnspacing=0.6,
)

add_labels(ax, r1, alignment='left', fs_scaled=fs_scaled)
add_labels(ax, r2, alignment='center', fs_scaled=fs_scaled)
add_labels(ax, r3, alignment='center', fs_scaled=fs_scaled)

plt.tight_layout()
fig.subplots_adjust(bottom=0.5)
plt.savefig(outdir + f'participation_Comparison_Models_{time_frame}yr_M{M_threshold}_{Model_Type}_TI.png',
            dpi=300, bbox_inches='tight', pad_inches=0.15)

plt.show()
