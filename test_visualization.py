#!/usr/bin/env python3
"""
Test script for TDOA and wave arrival visualizations
"""

from drone_localization_system import DroneLocalizationSystem
import matplotlib.pyplot as plt
import numpy as np

def test_visualizations():
    # Initialize the system
    system = DroneLocalizationSystem('config.json')

    print("Running simulation with pulse signals...")
    true_pos, est_pos, metrics = system.run_simulation()

    # Get the received signals for wave visualization
    print("Generating signals for visualization...")
    drone_signal = system.drone_signal_generator.generate_signal(
        system.duration, system.sample_rate
    )

    all_mic_positions = system.microphone_array.get_all_positions()
    all_received_signals = system.microphone_array.receive_signal(
        drone_signal, true_pos, system.sample_rate,
        system.speed_of_sound, system.noise_level
    )

    print("Creating visualizations...")

    # 1. Wave arrivals plot for each array
    print("Plotting wave arrivals...")
    for array_name, received_signals in all_received_signals.items():
        print(f"  - {array_name} wave arrivals")
        system.plot_wave_arrivals(received_signals, true_pos, array_name)

    # 2. TDOA analysis plot
    print("Plotting TDOA analysis...")
    system.plot_tdoa_analysis(true_pos, est_pos, metrics)

    # 3. Standard localization plot
    print("Plotting localization results...")
    system.plot_results(true_pos, est_pos)

    print("All visualizations complete!")

if __name__ == "__main__":
    test_visualizations()