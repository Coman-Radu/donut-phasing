#!/usr/bin/env python3
"""
Test script to analyze the effect of different TDOA/AOA fusion weights on localization precision.
"""

import numpy as np
from drone_localization_system import DroneLocalizationSystem


def test_fusion_weights():
    """Test different TDOA/AOA fusion weight combinations."""
    print("Testing TDOA/AOA Fusion Weight Effects on Localization Precision")
    print("=" * 80)

    # Test different weight combinations
    weight_combinations = [
        (0.5, 0.5),   # 50/50 - Equal weighting
        (0.7, 0.3),   # Current default (TDOA-heavy)
        (0.3, 0.7),   # AOA-heavy
        (0.9, 0.1),   # Almost pure TDOA
        (0.1, 0.9),   # Almost pure AOA
        (1.0, 0.0),   # Pure TDOA
        (0.0, 1.0),   # Pure AOA
    ]

    # Test positions (varying distances from array)
    test_positions = [
        [1.0, 1.0],   # Close
        [3.0, 3.0],   # Medium
        [5.0, 5.0],   # Far
        [9.0, 9.0],   # Very far
    ]

    results = {}

    for pos in test_positions:
        print(f"\n{'='*60}")
        print(f"TESTING POSITION: {pos}")
        print(f"{'='*60}")

        position_results = {}

        for tdoa_w, aoa_w in weight_combinations:
            print(f"\nWeights - TDOA: {tdoa_w:.1f}, AOA: {aoa_w:.1f}")

            # Run multiple trials to get statistical significance
            errors = []
            tdoa_errors = []
            aoa_errors = []

            for trial in range(5):
                system = DroneLocalizationSystem('config.json')
                system.update_drone_position(pos)

                # Modify the fusion weights
                original_method = system._combine_tdoa_aoa_estimates
                system._combine_tdoa_aoa_estimates = lambda tdoa_pos, aoa_pos, tdoa_m, aoa_m: \
                    custom_fusion(tdoa_pos, aoa_pos, tdoa_w, aoa_w)

                true_pos, estimated_pos, metrics = system.run_simulation()

                # Calculate errors
                combined_error = np.linalg.norm(estimated_pos - true_pos)
                tdoa_error = np.linalg.norm(np.array(metrics['tdoa_position']) - true_pos)
                aoa_error = np.linalg.norm(np.array(metrics['aoa_position']) - true_pos)

                errors.append(combined_error)
                tdoa_errors.append(tdoa_error)
                aoa_errors.append(aoa_error)

            # Calculate statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            mean_tdoa_error = np.mean(tdoa_errors)
            mean_aoa_error = np.mean(aoa_errors)

            position_results[f"TDOA_{tdoa_w:.1f}_AOA_{aoa_w:.1f}"] = {
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_tdoa_error': mean_tdoa_error,
                'mean_aoa_error': mean_aoa_error,
                'errors': errors
            }

            print(f"  Combined error: {mean_error:.3f} ± {std_error:.3f}m")
            print(f"  (TDOA only: {mean_tdoa_error:.3f}m, AOA only: {mean_aoa_error:.3f}m)")

        results[f"pos_{pos[0]}_{pos[1]}"] = position_results

    return results


def custom_fusion(tdoa_pos, aoa_pos, tdoa_weight, aoa_weight):
    """Custom fusion function with specified weights."""
    # Don't print to avoid cluttering output
    return tdoa_weight * tdoa_pos + aoa_weight * aoa_pos


def analyze_weight_sensitivity():
    """Analyze how sensitive the system is to weight changes."""
    print(f"\n{'='*80}")
    print("WEIGHT SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    # Fine-grained weight sweep around 50/50
    weights_to_test = [
        (0.4, 0.6), (0.45, 0.55), (0.5, 0.5), (0.55, 0.45), (0.6, 0.4)
    ]

    system = DroneLocalizationSystem('config.json')
    test_position = [3.0, 3.0]  # Medium distance
    system.update_drone_position(test_position)

    print(f"Testing position: {test_position}")
    print(f"Weight sensitivity (TDOA, AOA) -> Error")

    for tdoa_w, aoa_w in weights_to_test:
        # Run single trial for speed
        system._combine_tdoa_aoa_estimates = lambda tdoa_pos, aoa_pos, tdoa_m, aoa_m: \
            custom_fusion(tdoa_pos, aoa_pos, tdoa_w, aoa_w)

        true_pos, estimated_pos, metrics = system.run_simulation()
        error = np.linalg.norm(estimated_pos - true_pos)

        print(f"  ({tdoa_w:.2f}, {aoa_w:.2f}) -> {error:.3f}m")


def summarize_results(results):
    """Summarize the fusion weight test results."""
    print(f"\n{'='*80}")
    print("FUSION WEIGHT ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Find best weights for each position
    for pos_key, pos_results in results.items():
        pos_name = pos_key.replace('pos_', '').replace('_', ', ')
        print(f"\nPosition [{pos_name}]:")

        # Sort by mean error
        sorted_results = sorted(pos_results.items(), key=lambda x: x[1]['mean_error'])

        print("  Best to worst weight combinations:")
        for i, (weight_combo, metrics) in enumerate(sorted_results):
            weight_str = weight_combo.replace('TDOA_', '').replace('_AOA_', '/').replace('_', '')
            print(f"    {i+1}. {weight_str}: {metrics['mean_error']:.3f} ± {metrics['std_error']:.3f}m")

        # Highlight 50/50 performance
        fifty_fifty = pos_results.get('TDOA_0.5_AOA_0.5')
        if fifty_fifty:
            print(f"  -> 50/50 weighting: {fifty_fifty['mean_error']:.3f}m (rank: {sorted_results.index(('TDOA_0.5_AOA_0.5', fifty_fifty)) + 1})")


if __name__ == "__main__":
    # Test different fusion weights
    results = test_fusion_weights()

    # Analyze weight sensitivity
    analyze_weight_sensitivity()

    # Summarize findings
    summarize_results(results)

    print(f"\n{'='*80}")
    print("KEY INSIGHTS:")
    print("1. 50/50 weighting may not be optimal for all positions")
    print("2. TDOA generally more accurate for close positions")
    print("3. AOA may contribute more at specific frequencies/geometries")
    print("4. Optimal weights likely depend on distance and signal characteristics")
    print("5. Weight sensitivity shows how 'fragile' the fusion is")
    print(f"{'='*80}")