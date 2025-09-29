#!/usr/bin/env python3
"""
Test script to analyze the effect of different peak selection strategies on TDOA localization.
"""

import numpy as np
from drone_localization_system import DroneLocalizationSystem
from tdoa_localization import TDOALocalizer


def test_peak_analysis():
    """Test different peak selection strategies."""
    print("Testing Peak Selection Strategies for TDOA Localization")
    print("=" * 80)

    # Create system
    system = DroneLocalizationSystem('config.json')

    # Generate signals
    print("Generating drone signal...")
    true_pos, estimated_pos, metrics = system.run_simulation()

    print(f"\nTrue drone position: {true_pos}")
    print(f"Current system estimate: {estimated_pos}")

    # Get the received signals and mic positions for analysis
    drone_signal = system.drone_signal_generator.generate_signal(
        system.duration, system.sample_rate
    )
    received_signals = system.microphone_array.receive_signal(
        drone_signal,
        true_pos,
        system.config['simulation']['sample_rate'],
        system.config['simulation']['speed_of_sound'],
        system.config['simulation']['noise_level']
    )
    mic_positions = system.microphone_array.get_positions()

    # Create TDOA localizer for detailed analysis
    tdoa_localizer = TDOALocalizer(config_dict=system.config)

    print("\n" + "="*80)
    print("DETAILED CORRELATION ANALYSIS")
    print("="*80)

    # First, show detailed correlation analysis
    tdoa_localizer.localize(received_signals, mic_positions, 'analyze_all')

    print("\n" + "="*80)
    print("COMPARING DIFFERENT PEAK SELECTION METHODS")
    print("="*80)

    # Compare different methods
    comparison = tdoa_localizer.compare_peak_selection_methods(received_signals, mic_positions)

    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    print(f"True position: [{true_pos[0]:.3f}, {true_pos[1]:.3f}]")
    print()

    for method, results in comparison.items():
        pos = results['position']
        error = np.linalg.norm(np.array(pos) - true_pos)
        tdoa_stats = results['tdoa_summary']

        print(f"{method.upper().replace('_', ' ')}:")
        print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}]")
        print(f"  Error: {error:.3f} meters")
        print(f"  TDOA statistics:")
        print(f"    Mean |TDOA|: {tdoa_stats['mean']:.6f} seconds")
        print(f"    Std TDOA: {tdoa_stats['std']:.6f} seconds")
        print(f"    Range: [{tdoa_stats['range'][0]:.6f}, {tdoa_stats['range'][1]:.6f}] seconds")
        print()

    return comparison


def test_multiple_positions():
    """Test peak selection strategies at different drone positions."""
    print("\n" + "="*80)
    print("TESTING PEAK SELECTION ACROSS MULTIPLE POSITIONS")
    print("="*80)

    system = DroneLocalizationSystem('config.json')

    test_positions = [
        [1.0, 1.0],   # Close
        [3.0, 3.0],   # Medium
        [5.0, 5.0],   # Far
        [0.0, 2.0],   # Edge case
    ]

    all_results = {}

    for i, pos in enumerate(test_positions):
        print(f"\n--- Test Position {i+1}: {pos} ---")

        # Update drone position
        system.update_drone_position(pos)

        # Generate signals
        drone_signal = system.drone_signal_generator.generate_signal(
            system.duration, system.sample_rate
        )
        received_signals = system.microphone_array.receive_signal(
            drone_signal,
            np.array(pos),
            system.config['simulation']['sample_rate'],
            system.config['simulation']['speed_of_sound'],
            system.config['simulation']['noise_level']
        )
        mic_positions = system.microphone_array.get_positions()

        # Test different methods
        tdoa_localizer = TDOALocalizer(config_dict=system.config)
        comparison = tdoa_localizer.compare_peak_selection_methods(received_signals, mic_positions)

        # Calculate errors
        position_results = {}
        for method, results in comparison.items():
            estimated_pos = np.array(results['position'])
            error = np.linalg.norm(estimated_pos - np.array(pos))
            position_results[method] = {
                'position': results['position'],
                'error': error
            }

        all_results[f"pos_{i+1}_{pos[0]}_{pos[1]}"] = position_results

        # Print summary for this position
        print(f"Errors at position {pos}:")
        for method, result in position_results.items():
            print(f"  {method}: {result['error']:.3f}m")

    return all_results


if __name__ == "__main__":
    # Test peak analysis for single position
    single_test = test_peak_analysis()

    # Test across multiple positions
    multiple_test = test_multiple_positions()

    print("\n" + "="*80)
    print("OVERALL ANALYSIS COMPLETE")
    print("="*80)
    print("Key insights:")
    print("1. Multiple correlation peaks are found due to noise and signal characteristics")
    print("2. Different peak selection strategies can lead to significantly different TDOA estimates")
    print("3. The 'highest peak' method may not always choose the direct path")
    print("4. Geometric constraints ('expected_range') can help eliminate unrealistic TDOAs")
    print("5. First peak selection assumes direct path arrives first (may be better for multipath)")