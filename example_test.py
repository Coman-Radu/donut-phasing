#!/usr/bin/env python3
"""
Example test script for drone localization using TDOA and AOA with microphone arrays.

This script demonstrates:
1. Basic drone localization with default configuration (TDOA + AOA)
2. Testing different signal types (sine, complex harmonics)
3. Testing different drone positions
4. Testing with varying noise levels
5. Performance evaluation with GPS synchronization effects
6. Comparison between TDOA and AOA methods
"""

import numpy as np
import json
from drone_localization_system import DroneLocalizationSystem


def test_basic_localization():
    """Test basic drone localization with default configuration."""
    print("=" * 60)
    print("Test 1: Basic Drone Localization (TDOA + AOA Fusion)")
    print("=" * 60)

    # Create system with default configuration
    system = DroneLocalizationSystem('config.json')

    # Run simulation
    true_pos, estimated_pos, metrics = system.run_simulation()

    # Calculate errors for individual methods and combined result
    tdoa_error = np.linalg.norm(np.array(metrics['tdoa_position']) - true_pos)
    aoa_error = np.linalg.norm(np.array(metrics['aoa_position']) - true_pos)
    combined_error = np.linalg.norm(np.array(metrics['combined_position']) - true_pos)

    print(f"\nLocalization Results:")
    print(f"True position: {true_pos}")
    print(f"TDOA only: {metrics['tdoa_position']} (error: {tdoa_error:.3f}m)")
    print(f"AOA only: {metrics['aoa_position']} (error: {aoa_error:.3f}m)")
    print(f"Combined: {metrics['combined_position']} (error: {combined_error:.3f}m)")

    # Display fusion information
    print(f"\nFusion Method: {metrics['fusion_method']}")

    # Display AOA-specific metrics
    aoa_metrics = metrics['aoa_metrics']
    print(f"\nAOA Analysis:")
    print(f"Frequency used: {aoa_metrics['frequency_used']:.1f} Hz")
    print(f"Phase differences: {[f'{pd:.3f}' for pd in aoa_metrics['phase_differences']]}")
    print(f"Calculated angles: {[f'{a:.3f}' for a in aoa_metrics['angles']]}")

    # Plot results
    system.plot_results(true_pos, estimated_pos)

    # Print system info
    info = system.get_system_info()
    print(f"\nSystem Info: {json.dumps(info, indent=2)}")

    return true_pos, estimated_pos


def test_different_signal_types():
    """Test localization with different drone signal types."""
    print("=" * 60)
    print("Test 2: Different Signal Types")
    print("=" * 60)

    system = DroneLocalizationSystem('config.json')

    signal_types = ['sine', 'complex']
    results = {}

    for signal_type in signal_types:
        print(f"\nTesting with {signal_type} signal:")

        # Update configuration for current signal type
        new_config = system.config.copy()
        new_config['drone']['signal_type'] = signal_type
        system.update_config(new_config)

        # Run simulation
        true_pos, estimated_pos, metrics = system.run_simulation()

        combined_error = np.linalg.norm(np.array(metrics['combined_position']) - true_pos)
        tdoa_error = np.linalg.norm(np.array(metrics['tdoa_position']) - true_pos)
        aoa_error = np.linalg.norm(np.array(metrics['aoa_position']) - true_pos)

        results[signal_type] = {
            'true_position': true_pos.tolist(),
            'combined_position': metrics['combined_position'],
            'combined_error': combined_error,
            'tdoa_error': tdoa_error,
            'aoa_error': aoa_error,
            'aoa_frequency': metrics['aoa_metrics']['frequency_used']
        }

        # Plot results
        system.plot_results(true_pos, estimated_pos)

    print("\nSignal Type Comparison:")
    for signal_type, result in results.items():
        print(f"{signal_type}:")
        print(f"  Combined TDOA+AOA error: {result['combined_error']:.3f} meters")
        print(f"  (TDOA only: {result['tdoa_error']:.3f}m, AOA only: {result['aoa_error']:.3f}m)")
        print(f"  AOA frequency: {result['aoa_frequency']:.1f} Hz")

    return results


def test_multiple_positions():
    """Test localization accuracy at different drone positions."""
    print("=" * 60)
    print("Test 3: Multiple Drone Positions")
    print("=" * 60)

    system = DroneLocalizationSystem('config.json')

    # Test positions around the microphone arrays (including much further distances)
    test_positions = [
        [2.5, 1.5],   # Original position (close to Array 1)
        [1.0, 1.0],   # Inside Array 1
        [3.0, 0.0],   # To the right of Array 1
        [0.0, 3.0],   # Above Array 1
        [-1.0, -1.0], # Bottom left of Array 1
        [4.0, 4.0],   # Between arrays
        [9.0, 9.0],   # Close to Array 2
        [12.0, 12.0], # Beyond Array 2
        [5.0, 15.0],  # Far above both arrays
        [15.0, 5.0],  # Far to the right
        [-5.0, 5.0],  # Far to the left
        [5.0, -5.0],  # Far below Array 1
        [20.0, 20.0], # Very far position
        [25.0, 5.0],  # Very far horizontal
        [5.0, 25.0]   # Very far vertical
    ]

    results = []

    for i, pos in enumerate(test_positions):
        print(f"\nTest Position {i+1}: {pos}")

        # Update drone position
        system.update_drone_position(pos)

        # Run simulation
        true_pos, estimated_pos, metrics = system.run_simulation()

        error = np.linalg.norm(estimated_pos - true_pos)
        results.append({
            'true_position': true_pos.tolist(),
            'estimated_position': estimated_pos.tolist(),
            'error': error
        })

        # Plot results for first few positions
        if i < 3:
            system.plot_results(true_pos, estimated_pos)

    print("\nPosition Test Summary:")
    for i, result in enumerate(results):
        print(f"Position {i+1}: Error = {result['error']:.3f} meters")

    return results


def test_noise_sensitivity():
    """Test localization sensitivity to noise."""
    print("=" * 60)
    print("Test 4: Noise Sensitivity")
    print("=" * 60)

    system = DroneLocalizationSystem('config.json')

    noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
    results = []

    for noise in noise_levels:
        print(f"\nTesting with noise level: {noise}")

        # Update noise level
        new_config = system.config.copy()
        new_config['simulation']['noise_level'] = noise
        system.update_config(new_config)

        # Run multiple trials and average
        errors = []
        for trial in range(5):
            true_pos, estimated_pos, metrics = system.run_simulation()
            error = np.linalg.norm(estimated_pos - true_pos)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        results.append({
            'noise_level': noise,
            'mean_error': mean_error,
            'std_error': std_error,
            'errors': errors
        })

        print(f"Mean error: {mean_error:.3f} ± {std_error:.3f} meters")

    print("\nNoise Sensitivity Summary:")
    for result in results:
        print(f"Noise {result['noise_level']:.3f}: "
              f"Error = {result['mean_error']:.3f} ± {result['std_error']:.3f} m")

    return results


def test_gps_sync_effects():
    """Test the effects of GPS synchronization accuracy."""
    print("=" * 60)
    print("Test 5: GPS Synchronization Effects")
    print("=" * 60)

    system = DroneLocalizationSystem('config.json')

    # Test with GPS sync disabled vs enabled
    sync_configs = [
        {'enabled': False, 'clock_drift': 0, 'sync_accuracy': 0},
        {'enabled': True, 'clock_drift': 1e-9, 'sync_accuracy': 1e-8},
        {'enabled': True, 'clock_drift': 1e-8, 'sync_accuracy': 1e-7}
    ]

    results = []

    for i, sync_config in enumerate(sync_configs):
        config_name = f"GPS Config {i+1}"
        if not sync_config['enabled']:
            config_name = "No GPS Sync"

        print(f"\nTesting {config_name}:")
        print(f"Clock drift: {sync_config['clock_drift']}")
        print(f"Sync accuracy: {sync_config['sync_accuracy']}")

        # Update GPS sync configuration
        new_config = system.config.copy()
        new_config['microphones']['gps_sync'] = sync_config
        system.update_config(new_config)

        # Run multiple trials
        errors = []
        for trial in range(10):
            true_pos, estimated_pos, metrics = system.run_simulation()
            error = np.linalg.norm(estimated_pos - true_pos)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        results.append({
            'config_name': config_name,
            'gps_config': sync_config,
            'mean_error': mean_error,
            'std_error': std_error,
            'errors': errors
        })

        print(f"Mean error: {mean_error:.3f} ± {std_error:.3f} meters")

    print("\nGPS Synchronization Summary:")
    for result in results:
        print(f"{result['config_name']}: "
              f"Error = {result['mean_error']:.3f} ± {result['std_error']:.3f} m")

    return results


def main():
    """Run all tests."""
    print("Drone Localization System - Comprehensive Test Suite")
    print("=" * 80)

    try:
        # Run all tests
        test1_results = test_basic_localization()
        test2_results = test_different_signal_types()
        test3_results = test_multiple_positions()
        test4_results = test_noise_sensitivity()
        test5_results = test_gps_sync_effects()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("All tests completed. Check the plots and results above.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()