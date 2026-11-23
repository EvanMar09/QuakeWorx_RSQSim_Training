#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:44:19 2026

@author: emarschall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# --- Global Font Formatting ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

random_seed = 1
random.seed(random_seed)


# --- Configuration ---
indirs = [
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Connected_run_uniform3_low_fa/",
    "/Users/emarschall/Desktop/RSQSim_stuff/Bay_Area/New_RC_runs/Disconnected_run_uniform3_low_fa/",
]

outdir = "/Users/emarschall/Desktop/RSQSim_stuff/QuakeWorx_test/Stuff_4_Alice_prop/Figs/multiseg/"


filenames = [
    "connected_uni",
    "disconnected_uni",
]
fault_files = [
    "East_Bay_Updated_Connected_HSR3.flt",
    "East_Bay_Updated_Disconnected_HSR3.flt"
] 

Model_Type = "Uniform low_fa"
Model_names = ["Connected", "Disconnected"]

try:
    os.makedirs(outdir, exist_ok=True)
    print(f"Folder '{outdir}' exists.")
except OSError as e:
    print(f"Error creating folder: {e}")

M_threshold = 6.5
beginning_string = "%%% end input files"
col_names = ["t","Mo","Mw","nuc_x","nuc_y","nuc_z","area","duration","nuc_element"]

# Dictionaries to store raw segment counts
raw_counts = {model: {} for model in Model_names}
fault_strings = None 

# --- Process Both Models ---
for m_idx, model_name in enumerate(Model_names):
    print(f"Processing model: {model_name}...")
    
    indir = indirs[m_idx]
    filename = filenames[m_idx]
    infile = "eqs." + filename + ".out"
    e_file = filename + ".eList"
    p_file = filename + ".pList"
    
    filepath = indir + infile
    fault_filepath = indir + fault_files[m_idx]
    
    # Locate data start
    starting_index = []
    with open(filepath, 'r') as file:
        for line_num, line in enumerate(file, 0):
            if beginning_string in line:
                starting_index.append(line_num)
                break
                
    # Load events
    df_events = pd.read_csv(filepath, sep=" ", header=None, skiprows=starting_index[0]+1, names=col_names)
    df_events.t = df_events.t / (60*60*24*365.25) 
    
    # Load patch and event lists
    data_events = np.fromfile(indir + e_file, dtype=np.int32)
    data_patches = np.fromfile(indir + p_file, dtype=np.int32)
    
    time_skip = 1 if "Creeping" in Model_Type else 5000
    
    # Load fault file
    fault = pd.read_csv(fault_filepath, sep=" ", header=None)
    if fault_strings is None:
        fault_strings = np.unique(fault[12]).tolist()
        
    # Initialize dictionary structure for this model
    raw_counts[model_name] = {fault_name: [] for fault_name in fault_strings}

    valid_events_df = df_events[(df_events.Mw >= M_threshold) & (df_events.t >= time_skip)]
    valid_event_ids = valid_events_df.index.values + 1
    
    for event in valid_event_ids:
        # Get nucleation fault
        nuc_element = df_events.nuc_element[event-1]
        nuc_fault = fault[12][nuc_element]
        
        # We only care about faults in our unified list
        if nuc_fault in fault_strings:
            event_indices = np.flatnonzero(np.isin(data_events, event))
            patches = data_patches[event_indices]
            number_segments = len(np.unique(fault[12][patches-1]))
            
            raw_counts[model_name][nuc_fault].append(number_segments)

# --- Binary Normalization into Percentages ---
segment_categories = ["1 Segment", "Multi-Segment"]
results = {model: {seg: [] for seg in segment_categories} for model in Model_names}

for model_name in Model_names:
    for fault_name in fault_strings:
        total_fault_events = len(raw_counts[model_name][fault_name])
        if total_fault_events > 0:
            single_count = sum(1 for seg in raw_counts[model_name][fault_name] if seg == 1)
            multi_count = sum(1 for seg in raw_counts[model_name][fault_name] if seg > 1)
            
            single_pct = (single_count / total_fault_events) * 100
            multi_pct = (multi_count / total_fault_events) * 100
        else:
            single_pct = 0.0
            multi_pct = 0.0
            
        results[model_name]["1 Segment"].append(single_pct)
        results[model_name]["Multi-Segment"].append(multi_pct)


#%% --- Plotting Combined Models ---
x = np.arange(len(fault_strings))
num_models = len(Model_names)
num_segs = len(segment_categories)
total_bars = num_models * num_segs  # Total of 4 bars per fault group

# Wider individual bars because we only have 4 bars per fault group
width = 0.95 / total_bars 

fig, ax = plt.subplots(figsize=(6, 2))

# Unified Font Size Configuration
fs_scaled = 6.5

# Yellow and Red colors for Binary comparison (1 Segment vs Multi-Segment)
colors = ['#FFEDA0', '#E31A1C']

bar_rects = []

for idx, seg in enumerate(segment_categories):
    color = colors[idx]
    
    # 1. Connected Models (Solid)
    offset_conn = (((idx * 2) - (total_bars - 1) / 2) * width)
    rects_conn = ax.bar(x + offset_conn, results['Connected'][seg], width, 
                        label=f'Connected ({seg})', color=color, edgecolor='black', alpha=0.95, linewidth=0.5)
    bar_rects.append(rects_conn)
    
    # 2. Disconnected Models (Hatched '//')
    offset_disconn = (((idx * 2 + 1) - (total_bars - 1) / 2) * width)
    rects_disconn = ax.bar(x + offset_disconn, results['Disconnected'][seg], width, 
                           label=f'Disconnected ({seg})', color=color, edgecolor='black', hatch='//', alpha=0.95, linewidth=0.5)
    bar_rects.append(rects_disconn)

# Formatting
ax.set_ylabel('Ruptures (%)', fontsize=fs_scaled, fontweight='bold')
ax.set_title(f'Single vs. Multi-Segment Ruptures (M $\\geq$ {M_threshold}) For {Model_Type}: Connected vs. Disconnected', fontsize=fs_scaled, fontweight='bold', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(fault_strings, rotation=35, ha='right', fontsize=fs_scaled)
ax.tick_params(axis='both', labelsize=fs_scaled)
ax.set_xlabel('Nucleation Fault Segment', fontsize=fs_scaled, fontweight='bold', labelpad=0)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# Scale y-limit smoothly up to 100% max
max_val = max([max(results[m][s]) for m in Model_names for s in segment_categories if results[m][s]])
ax.set_ylim(0, min(110, max_val + 18)) 

# Draw the full box (top and right edges too) and leave a little room on each side
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')
ax.margins(x=0.02)

# Position legend beneath the figure
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.55), 
    fontsize=fs_scaled - 1.0, 
    ncol=4, 
    frameon=True,
    framealpha=0.9, 
    handletextpad=0.3, 
    columnspacing=1.0
)

# Horizontal labels centered above each bar
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        if pd.isna(height) or height <= 0:
            continue
        
        x_anchor = rect.get_x() + (rect.get_width() / 2)
        ax.annotate(f'{height:.1f}%',
                     xy=(x_anchor, height),
                     xytext=(0, 0), textcoords="offset points",
                     ha='center', va='bottom', fontsize=fs_scaled - 1.8, fontweight='semibold', rotation=0) 

# Apply labels to all generated bars
for rects in bar_rects:
    add_labels(rects)

# Tight layout with bottom clearance
plt.tight_layout()
fig.subplots_adjust(bottom=0.4)

plt.savefig(outdir + f'Comparison_Binary_Segments_{Model_Type}_{M_threshold}.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.show()