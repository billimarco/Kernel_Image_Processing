import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Parametri
kernel_types = ['gaussian']
kernel_sizes = ['3x3', '7x7', '11x11']
base_path = 'build/Release/Debug/results'           # cartella con csv

labels = ['(4,4)', '(8,4)', '(8,8)', '(16,8)', '(16,16)', '(32,8)', '(32,16)', '(32,32)']

fig, axes = plt.subplots(2, len(kernel_sizes), figsize=(11.69, 8.27), sharex=True, sharey='row')

for j, size in enumerate(kernel_sizes):
    speedup_ax = axes[0, j]
    time_ax = axes[1, j]
    
    file_path_time = os.path.join(base_path, f"{kernel_types[0]}_{size}", f"csv_times_{kernel_types[0]}_{size}.csv")
    file_path_speedup = os.path.join(base_path, f"{kernel_types[0]}_{size}", f"csv_speedup_{kernel_types[0]}_{size}.csv")
    
    if not os.path.exists(file_path_time):
        print(f"File tempi mancante: {file_path_time}")
        continue
    if not os.path.exists(file_path_speedup):
        print(f"File speedup mancante: {file_path_speedup}")
        continue

    df_time = pd.read_csv(file_path_time, sep=';')
    df_time.columns = df_time.columns.str.strip()

    df_speedup = pd.read_csv(file_path_speedup, sep=';')
    df_speedup.columns = df_speedup.columns.str.strip()

    xs = []
    time_conv = []
    time_sep = []
    speedup_conv = []
    speedup_sep = []

    for label in labels:
        if label in df_time['Block_Size'].values and label in df_speedup['Block_Size'].values:
            row_time = df_time[df_time['Block_Size'] == label]
            row_speedup = df_speedup[df_speedup['Block_Size'] == label]

            time_conv.append(row_time['Convolution'].values[0])
            time_sep.append(row_time['Separable'].values[0])
            speedup_conv.append(row_speedup['Convolution'].values[0])
            speedup_sep.append(row_speedup['Separable'].values[0])
            xs.append(label)

    x = np.arange(len(xs))
    width = 0.35

    speedup_ax.bar(x - width/2, speedup_conv, width, color='tab:blue', alpha=0.7, label='Convolution')
    speedup_ax.bar(x + width/2, speedup_sep, width, color='tab:orange', alpha=0.7, label='Separable')
    speedup_ax.set_xticks(x)
    speedup_ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=9)
    speedup_ax.set_title(f"Kernel {size} - Speedup", fontsize=11)
    speedup_ax.grid(True)
    if j == 0:
        speedup_ax.set_ylabel("Speedup", fontsize=10)
        speedup_ax.yaxis.label.set_fontsize(10)

    time_ax.bar(x - width/2, time_conv, width, color='tab:blue', label='Convolution')
    time_ax.bar(x + width/2, time_sep, width, color='tab:orange', alpha=0.7, label='Separable')
    time_ax.set_xticks(x)
    time_ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=9)
    time_ax.set_title(f"Kernel {size} - Tempi", fontsize=11)
    time_ax.grid(True)
    if j == 0:
        time_ax.set_ylabel("Tempo (ms)", fontsize=10)
        time_ax.yaxis.label.set_fontsize(10)
        
handles, labels_ = axes[0,0].get_legend_handles_labels()
handles2, labels2 = axes[1,0].get_legend_handles_labels()
all_handles = handles + handles2
all_labels = labels_ + labels2
from collections import OrderedDict
by_label = OrderedDict(zip(all_labels, all_handles))

fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.955))

plt.suptitle("Speedup and Time Comparison for Gaussian Blur", fontsize=14)
plt.tight_layout(rect=[0, 0, 1.00, 0.95])
plt.show()

fig_seq, ax_seq = plt.subplots(1, 1, figsize=(8, 5))

x_labels = []
conv_times = []
sep_times = []

for size in kernel_sizes:
    file_path_time = os.path.join(base_path, f"{kernel_types[0]}_{size}", f"csv_times_{kernel_types[0]}_{size}.csv")
    if not os.path.exists(file_path_time):
        continue
    df_time = pd.read_csv(file_path_time, sep=';')
    df_time.columns = df_time.columns.str.strip()
    seq_row = df_time[df_time['Block_Size'] == 'SEQ']
    if seq_row.empty:
        continue
    seq_conv = seq_row['Convolution'].values[0]
    seq_sep = seq_row['Separable'].values[0]

    x_labels.append(size)
    conv_times.append(seq_conv)
    sep_times.append(seq_sep)

x = np.arange(len(x_labels))
width = 0.35

ax_seq.bar(x - width/2, conv_times, width, color='tab:blue', alpha=0.7, label='Convolution')
ax_seq.bar(x + width/2, sep_times, width, color='tab:orange', alpha=0.7, label='Separable')

ax_seq.set_xticks(x)
ax_seq.set_xticklabels(x_labels, rotation=45, fontsize=9)
ax_seq.legend(fontsize=12)

ax_seq.set_title("Sequential times for Gaussian Blur", fontsize=14)
ax_seq.set_ylabel("Tempo (ms)", fontsize=10)
ax_seq.grid(True)
ax_seq.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.show()

