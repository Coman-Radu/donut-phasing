import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Parameters
frequency = 0.5  # Hz (0.5 Hz = 2 second period)
duration = 20  # seconds - 20 second continuous signal
sampling_rate = 100  # Hz
time_delay = 5  # seconds
amplitude = 1

# Create time array
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Create first sine wave (reference)
signal1 = amplitude * np.sin(2 * np.pi * frequency * t)

# Create second sine wave aftre a 5-second delay
# This means signal2 starts 5 seconds after signal1 would start
signal2 = np.zeros_like(signal1)
delay_samples = int(time_delay * sampling_rate)
if delay_samples < len(signal1):
    signal2[delay_samples:] = amplitude * np.sin(2 * np.pi * frequency * t[:-delay_samples])

# Perform cross-correlation
correlation = correlate(signal1, signal2, mode='full')

# Create lag array for correlation plot
lags = np.arange(-len(signal1) + 1, len(signal1))
lag_time = lags / sampling_rate

# Find peak correlation and its lag
peak_idx = np.argmax(correlation)
peak_lag = lag_time[peak_idx]
peak_value = correlation[peak_idx]

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot original signals
axes[0].plot(t, signal1, 'b-', label='Signal 1 (Reference)', linewidth=2)
axes[0].plot(t, signal2, 'r-', label='Signal 2 (Delayed by 5s)', linewidth=2)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Sine Waves: Reference vs Delayed')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 15)  # Show first 15 seconds for clarity

# Plot cross-correlation
axes[1].plot(lag_time, correlation, 'g-', linewidth=2)
axes[1].axvline(x=peak_lag, color='red', linestyle='--',
                label=f'Peak at {peak_lag:.2f}s lag')
axes[1].axvline(x=time_delay, color='orange', linestyle='--',
                label=f'True delay: {time_delay}s')
axes[1].set_xlabel('Lag (s)')
axes[1].set_ylabel('Cross-correlation')
axes[1].set_title('Cross-Correlation Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-10, 10)  # Focus on relevant lag range

# Plot zoomed correlation around peak
zoom_range = 2  # seconds around peak
zoom_mask = (lag_time >= peak_lag - zoom_range) & (lag_time <= peak_lag + zoom_range)
axes[2].plot(lag_time[zoom_mask], correlation[zoom_mask], 'g-', linewidth=2)
axes[2].axvline(x=peak_lag, color='red', linestyle='--',
                label=f'Detected delay: {peak_lag:.2f}s')
axes[2].axvline(x=time_delay, color='orange', linestyle='--',
                label=f'True delay: {time_delay}s')
axes[2].set_xlabel('Lag (s)')
axes[2].set_ylabel('Cross-correlation')
axes[2].set_title('Cross-Correlation (Zoomed around Peak)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results
print(f"Signal parameters:")
print(f"  Frequency: {frequency} Hz")
print(f"  Period: {1/frequency} seconds")
print(f"  Duration: {duration} seconds")
print(f"  True time delay: {time_delay} seconds")
print(f"  Sampling rate: {sampling_rate} Hz")
print(f"  Number of samples: {len(t)}")
print(f"\nCross-correlation results:")
print(f"  Peak correlation value: {peak_value:.3f}")
print(f"  Detected time delay: {peak_lag:.3f} seconds")
print(f"  Error: {abs(peak_lag - time_delay):.3f} seconds")