import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('./data/anonymised/york/dl/4_bxb.csv', sep=',')
columns_to_plot = ['VO2_kg mL/kg/min', 'Work_Watts Watts', 'Vd_est mL', 'VCO2_BSA mL/m^2', 'HRR %', 'Ti sec']

# Create a new DataFrame with selected columns
plot_data = data[columns_to_plot]

# Calculate breath numbers for x-axis
breath_numbers = np.arange(1, len(data) + 1)

# Plotting
fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(15, 3*len(columns_to_plot)), sharex=True)
fig.suptitle('CPET Data Visualization')

for i, (ax, col) in enumerate(zip(axes, columns_to_plot)):
    ax.plot(breath_numbers, plot_data[col], linewidth=1)
    ax.set_ylabel(col)
    ax.grid(True, linestyle='--', alpha=0.7)

axes[-1].set_xlabel('Breath Number')
plt.tight_layout()
plt.show()

# Print range of breath numbers for reference
print(f"Breath number range: 1 to {len(data)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import dct

# Read the data
data = pd.read_csv('./data/anonymised/york/dl/4_bxb.csv', sep=',')
columns_to_plot = ['VO2_kg mL/kg/min', 'Work_Watts Watts', 'Vd_est mL','VCO2_BSA mL/m^2', 'HRR %', 'Ti sec']

# Create a new DataFrame with selected columns
plot_data = data[columns_to_plot]

# Normalize the data
normalized_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())

# Function to compute MFCC-like coefficients
def compute_mfcc_like(signal, n_mfcc=13):
    frame_length = 50
    frame_step = 25
    frames = [signal[i:i+frame_length] for i in range(0, len(signal) - frame_length + 1, frame_step)]
    frames = [np.hamming(frame_length) * frame for frame in frames]
    power_frames = [np.abs(np.fft.rfft(frame))**2 for frame in frames]
    mfcc = [dct(frame, type=2, norm='ortho')[:n_mfcc] for frame in power_frames]
    return np.array(mfcc).T

# Compute MFCC-like coefficients for each variable
mfcc_like_data = [compute_mfcc_like(normalized_data[col].values) for col in columns_to_plot]

# Normalize MFCC-like coefficients between 0 and 1 for each variable
normalized_mfcc_data = []
for mfcc in mfcc_like_data:
    mfcc_min = mfcc.min()
    mfcc_max = mfcc.max()
    normalized_mfcc = (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
    normalized_mfcc_data.append(normalized_mfcc)

# Calculate breath numbers for x-axis
frame_length = 50
frame_step = 25
total_frames = normalized_mfcc_data[0].shape[1]
breath_numbers = np.linspace(frame_length//2, len(data) - frame_length//2, total_frames)

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle('CPET Data MFCC-like Visualization (Normalized)')

# Combine all normalized MFCC data into a single array
combined_mfcc = np.vstack(normalized_mfcc_data)

# Create the heatmap
im = ax.imshow(combined_mfcc, aspect='auto', origin='lower', cmap='viridis', 
               extent=[breath_numbers[0], breath_numbers[-1], 0, combined_mfcc.shape[0]],
               vmin=0, vmax=1)  # Set color scale limits

# Set y-axis ticks and labels
y_ticks = np.cumsum([mfcc.shape[0] for mfcc in normalized_mfcc_data]) - 0.5
ax.set_yticks(y_ticks)
ax.set_yticklabels(columns_to_plot)

ax.set_xlabel('Breath Number')
ax.set_ylabel('Variables')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Normalized MFCC-like Coefficient Magnitude')

plt.tight_layout()
plt.show()

# Print range of breath numbers for reference
print(f"Breath number range: {breath_numbers[0]:.0f} to {breath_numbers[-1]:.0f}")

<<<<<<< HEAD
=======

# Normalize the data
normalized_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())

# Function to compute MFCC-like coefficients without Hamming window
def compute_mfcc_like(signal, n_mfcc=13):
    frame_length = 50
    frame_step = 25
    frames = [signal[i:i+frame_length] for i in range(0, len(signal) - frame_length + 1, frame_step)]
    power_frames = [np.abs(np.fft.rfft(frame))**2 for frame in frames]
    mfcc = [dct(frame, type=2, norm='ortho')[:n_mfcc] for frame in power_frames]
    return np.array(mfcc).T

# Compute MFCC-like coefficients for each variable
mfcc_like_data = [compute_mfcc_like(normalized_data[col].values) for col in columns_to_plot]

# Normalize MFCC-like coefficients between 0 and 1 for each variable
normalized_mfcc_data = []
for mfcc in mfcc_like_data:
    mfcc_min = mfcc.min()
    mfcc_max = mfcc.max()
    normalized_mfcc = (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
    normalized_mfcc_data.append(normalized_mfcc)

# Calculate breath numbers for x-axis
frame_length = 50
frame_step = 25
total_frames = normalized_mfcc_data[0].shape[1]
breath_numbers = np.linspace(frame_length//2, len(data) - frame_length//2, total_frames)

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle('CPET Data MFCC-like Visualization (Normalized, No Hamming Window)')

# Combine all normalized MFCC data into a single array
combined_mfcc = np.vstack(normalized_mfcc_data)

# Create the heatmap
im = ax.imshow(combined_mfcc, aspect='auto', origin='lower', cmap='viridis', 
               extent=[breath_numbers[0], breath_numbers[-1], 0, combined_mfcc.shape[0]],
               vmin=0, vmax=1)  # Set color scale limits

# Set y-axis ticks and labels
y_ticks = np.cumsum([mfcc.shape[0] for mfcc in normalized_mfcc_data]) - 0.5
ax.set_yticks(y_ticks)
ax.set_yticklabels(columns_to_plot)

ax.set_xlabel('Breath Number')
ax.set_ylabel('Variables')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Normalized MFCC-like Coefficient Magnitude')

plt.tight_layout()
plt.show()

# Print range of breath numbers for reference
print(f"Breath number range: {breath_numbers[0]:.0f} to {breath_numbers[-1]:.0f}")
>>>>>>> 9026843bef133ac4c5dbebda88b74cf20573a24d
