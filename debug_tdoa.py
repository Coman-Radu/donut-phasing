#!/usr/bin/env python3
"""
Debug TDOA calculation by examining actual signals
"""

import numpy as np
import matplotlib.pyplot as plt
from drone_localization_system import DroneLocalizationSystem

# Create system
config = {
  "microphones": {
    "array_1": {
      "positions": [
        [0.0, 0.0],
        [50.0, 0.0],
        [25.0, 43.3]
      ]
    },
    "array_2": {
      "positions": [
        [200.0, 0.0],
        [250.0, 0.0],
        [225.0, 43.3]
      ]
    },
    "sensitivity": -40.0,
    "frequency_response": {
      "low_cutoff": 20.0,
      "high_cutoff": 20000.0
    },
    "dynamic_range": 120.0,
    "directionality": {
      "type": "omnidirectional",
      "pattern_coefficient": 0.0
    },
    "audio_transparency": 0.95,
    "gps_sync": {
      "enabled": True,
      "clock_drift": 1e-9,
      "sync_accuracy": 1e-8
    }
  },
  "drone": {
    "position": [100.0, 100.0],
    "signal_type": "sine",
    "signal_config": {
      "sine": {
        "frequency": 1000.0,
        "amplitude": 1.0,
        "pulse_width": 0.2,
        "pulse_interval": 0.21
      }
    }
  },
  "simulation": {
    "sample_rate": 44100,
    "duration": 0.2,  # Short duration, covers 2 pulses
    "noise_level": 0.0,  # No noise for debugging
    "speed_of_sound": 343.0
  }
}

# Save debug config
import json
with open('debug_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("TDOA Debug Analysis")
print("=" * 50)

system = DroneLocalizationSystem('debug_config.json')

# Generate signal and get received signals
drone_signal = system.drone_signal_generator.generate_signal(
    config['simulation']['duration'],
    config['simulation']['sample_rate']
)

true_drone_position = np.array(config['drone']['position'])
all_mic_positions = system.microphone_array.get_all_positions()

print(f"True drone position: {true_drone_position}")
print(f"Signal length: {len(drone_signal)} samples ({len(drone_signal)/config['simulation']['sample_rate']:.3f}s)")

# Get received signals
received_signals = system.microphone_array.receive_signal(
    drone_signal, true_drone_position,
    config['simulation']['sample_rate'],
    config['simulation']['speed_of_sound'],
    config['simulation']['noise_level']
)

# Analyze first array signals
array1_signals = received_signals['array_1']
array1_positions = all_mic_positions['array_1']

print("\nArray 1 analysis:")
print("Microphone positions:")
for i, pos in enumerate(array1_positions):
    print(f"  Mic {i}: {pos}")

# Calculate theoretical arrival times
speed_of_sound = config['simulation']['speed_of_sound']
theoretical_delays = []
print("\nTheoretical propagation delays (relative to mic 0):")
for i, pos in enumerate(array1_positions):
    distance = np.linalg.norm(true_drone_position - pos)
    delay = distance / speed_of_sound
    if i == 0:
        ref_delay = delay
    theoretical_delays.append(delay - ref_delay)
    print(f"  Mic {i}: distance={distance:.1f}m, delay={delay:.6f}s, relative={delay-ref_delay:.6f}s")

print(f"\nTheoretical TDOAs: {theoretical_delays[1:]}")

# Calculate measured TDOAs
localizer = system.localizer
measured_tdoas = []
measured_weights = []

print("\nMeasured TDOAs:")
for i in range(1, len(array1_signals)):
    tdoa, weight = localizer.cross_correlation_tdoa(array1_signals[0], array1_signals[i])
    measured_tdoas.append(tdoa)
    measured_weights.append(weight)
    print(f"  Mic 0->Mic {i}: TDOA={tdoa:.6f}s, weight={weight:.1f}")

print(f"\nMeasured TDOAs: {measured_tdoas}")
print(f"Theoretical TDOAs: {theoretical_delays[1:]}")
print(f"Differences: {[m - t for m, t in zip(measured_tdoas, theoretical_delays[1:])]}")

# Plot the signals to visualize
plt.figure(figsize=(15, 10))

# Plot 1: Original drone signal
plt.subplot(3, 2, 1)
t = np.linspace(0, config['simulation']['duration'], len(drone_signal))
plt.plot(t, drone_signal)
plt.title('Original Drone Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot 2-4: Received signals at different mics
for i in range(min(3, len(array1_signals))):
    plt.subplot(3, 2, i+2)
    plt.plot(t, array1_signals[i])
    plt.title(f'Received at Mic {i} - Position {array1_positions[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Plot 5: Cross-correlation between mic 0 and mic 1
plt.subplot(3, 2, 5)
correlation = np.correlate(array1_signals[1], array1_signals[0], mode='full')
correlation_time = np.arange(-len(array1_signals[0])+1, len(array1_signals[0])) / config['simulation']['sample_rate']
plt.plot(correlation_time, np.abs(correlation))
plt.title('Cross-correlation |Mic 0 vs Mic 1|')
plt.xlabel('Time Delay (s)')
plt.ylabel('Correlation')
plt.grid(True)

# Mark the peak
peak_idx = np.argmax(np.abs(correlation))
peak_delay = correlation_time[peak_idx]
plt.axvline(peak_delay, color='red', linestyle='--', label=f'Peak at {peak_delay:.6f}s')
plt.axvline(theoretical_delays[1], color='green', linestyle='--', label=f'Expected at {theoretical_delays[1]:.6f}s')
plt.legend()

# Plot 6: Signal zoom around first pulse
plt.subplot(3, 2, 6)
zoom_samples = int(0.03 * config['simulation']['sample_rate'])  # Show first 30ms
t_zoom = t[:zoom_samples]
plt.plot(t_zoom, array1_signals[0][:zoom_samples], label='Mic 0', alpha=0.7)
plt.plot(t_zoom, array1_signals[1][:zoom_samples], label='Mic 1', alpha=0.7)
if len(array1_signals) > 2:
    plt.plot(t_zoom, array1_signals[2][:zoom_samples], label='Mic 2', alpha=0.7)
plt.title('First Pulse Arrival (Zoomed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nDebugging complete. Check the plots for signal timing analysis.")