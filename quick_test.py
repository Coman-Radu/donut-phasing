#!/usr/bin/env python3
"""
Quick test to debug Y-axis bounds issue
"""

import numpy as np
import json
from drone_localization_system import DroneLocalizationSystem

# Create a quick config for testing with sine wave (no WAV file)
quick_config = {
  "microphones": {
    "array_1": {
      "positions": [
        [0.0, 0.0],
        [50.0, 0.0],
        [25.0, 43.3],
        [25.0, -43.3]
      ]
    },
    "array_2": {
      "positions": [
        [200.0, 0.0],
        [250.0, 0.0],
        [225.0, 43.3],
        [225.0, -43.3]
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
    "position": [200.0, 150.0],
    "signal_type": "sine",
    "signal_config": {
      "sine": {
        "frequency": 1000.0,
        "amplitude": 1.0,
        "pulse_width": 0.01,
        "pulse_interval": 0.1
      }
    }
  },
  "simulation": {
    "sample_rate": 44100,
    "duration": 1.0,  # Short duration for quick testing
    "noise_level": 0.01,
    "speed_of_sound": 343.0
  }
}

# Save quick config
with open('quick_config.json', 'w') as f:
    json.dump(quick_config, f, indent=2)

print("Quick test for Y-axis bounds debugging")
print("=" * 50)

# Create system with quick config
system = DroneLocalizationSystem('quick_config.json')

# Run simulation and check TDOAs
true_pos, estimated_pos, metrics = system.run_simulation()

print(f"True position: {true_pos}")
print(f"Estimated position: {estimated_pos}")
print(f"Error: {np.linalg.norm(estimated_pos - true_pos):.1f}m")
print()

# Print TDOA details
if 'all_tdoas' in metrics:
    print("TDOA details:")
    for key, values in metrics['all_tdoas'].items():
        print(f"  {key}: {values}")

if 'all_weights' in metrics:
    print("Weight details:")
    for key, values in metrics['all_weights'].items():
        print(f"  {key}: {values}")

# Check if hitting bounds
if abs(estimated_pos[1] - 343.3) < 0.1 or abs(estimated_pos[1] + 343.3) < 0.1:
    print("\nWARNING: Hitting Y-axis bounds!")

    # Let's check theoretical TDOAs for the true position
    all_mic_positions = system.microphone_array.get_all_positions()
    localizer = system.localizer

    print("\nTheoretical TDOAs for true position:")
    for array_name, positions in all_mic_positions.items():
        theoretical_tdoas = localizer.calculate_theoretical_tdoas(true_pos, positions)
        print(f"  {array_name}: {theoretical_tdoas}")