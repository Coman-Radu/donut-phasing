import numpy as np
from scipy import signal
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import json


class TDOALocalizer:
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = config_dict

        self.speed_of_sound = config['simulation']['speed_of_sound']
        self.sample_rate = config['simulation']['sample_rate']

    def cross_correlation_tdoa(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[float, float]:
        """Calculate TDOA between two signals using cross-correlation."""
        correlation = signal.correlate(signal2, signal1, mode='full')

        # Find the peak correlation
        peak_index = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_index])

        # Calculate time delay in samples
        delay_samples = peak_index - (len(signal1) - 1)

        # Convert to time delay in seconds
        time_delay = delay_samples / self.sample_rate

        # Calculate correlation quality as weight (higher correlation = higher weight)
        max_correlation = np.max(np.abs(correlation))
        mean_correlation = np.mean(np.abs(correlation))
        weight = max_correlation / (mean_correlation + 1e-10)  # Avoid division by zero

        return time_delay, weight

    def calculate_all_tdoas(self, received_signals: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Calculate TDOAs within each array and between arrays with weights."""
        all_tdoas = {}
        all_weights = {}

        # Calculate TDOAs within each array
        for array_name, signals in received_signals.items():
            n_mics = len(signals)
            tdoas = []
            weights = []

            # Use first microphone as reference within each array
            for i in range(1, n_mics):
                # Apply windowing to reduce edge effects for pulsed signals
                window = np.hanning(len(signals[0]))
                sig1_windowed = signals[0] * window
                sig2_windowed = signals[i] * window

                tdoa, weight = self.cross_correlation_tdoa(sig1_windowed, sig2_windowed)
                tdoas.append(tdoa)
                weights.append(weight)

            all_tdoas[f'{array_name}_internal'] = np.array(tdoas)
            all_weights[f'{array_name}_internal'] = np.array(weights)

        # Calculate cross-array TDOAs (between array centers)
        if 'array_1' in received_signals and 'array_2' in received_signals:
            cross_array_tdoas = []
            cross_array_weights = []
            array1_signals = received_signals['array_1']
            array2_signals = received_signals['array_2']

            # Cross-correlate corresponding microphones between arrays
            for i in range(min(len(array1_signals), len(array2_signals))):
                window = np.hanning(len(array1_signals[0]))
                sig1_windowed = array1_signals[i] * window
                sig2_windowed = array2_signals[i] * window

                tdoa, weight = self.cross_correlation_tdoa(sig1_windowed, sig2_windowed)
                cross_array_tdoas.append(tdoa)
                cross_array_weights.append(weight)

            all_tdoas['cross_array'] = np.array(cross_array_tdoas)
            all_weights['cross_array'] = np.array(cross_array_weights)

        return all_tdoas, all_weights

    def hyperbola_intersection_2d(self, mic_positions: np.ndarray, tdoas: np.ndarray) -> np.ndarray:
        """Solve for source position using hyperbola intersection method."""

        def objective_function(pos):
            x, y = pos
            source_pos = np.array([x, y])

            error = 0
            # Calculate theoretical TDOAs and compare with measured
            for i in range(1, len(mic_positions)):
                # Distance from source to reference mic (mic 0)
                d0 = np.linalg.norm(source_pos - mic_positions[0])
                # Distance from source to mic i
                di = np.linalg.norm(source_pos - mic_positions[i])

                # Theoretical TDOA (time difference)
                theoretical_tdoa = (di - d0) / self.speed_of_sound

                # Error between measured and theoretical TDOA
                error += (tdoas[i-1] - theoretical_tdoa) ** 2

            return error

        # Initial guess - centroid of microphones
        initial_guess = np.mean(mic_positions, axis=0)

        # Optimization bounds (reasonable search area)
        mic_extent = np.max(mic_positions) - np.min(mic_positions)
        bounds = [
            (np.min(mic_positions[:, 0]) - mic_extent, np.max(mic_positions[:, 0]) + mic_extent),
            (np.min(mic_positions[:, 1]) - mic_extent, np.max(mic_positions[:, 1]) + mic_extent)
        ]

        # Solve optimization problem
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            return result.x
        else:
            print(f"Warning: Optimization failed. Using initial guess.")
            return initial_guess

    def localize(self, received_signals: Dict[str, List[np.ndarray]], mic_positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """Main localization function with dual array support and weighted least squares."""

        # Calculate TDOAs and weights for all arrays
        all_tdoas, all_weights = self.calculate_all_tdoas(received_signals)

        # Use combined approach: internal array TDOAs + cross-array baseline
        if 'cross_array' in all_tdoas:
            # Multi-array weighted localization
            estimated_position = self.multi_array_weighted_localization(mic_positions, all_tdoas, all_weights)
        else:
            # Single array fallback
            array_name = list(received_signals.keys())[0]
            tdoas = all_tdoas[f'{array_name}_internal']
            estimated_position = self.hyperbola_intersection_2d(mic_positions[array_name], tdoas)

        # Calculate quality metrics
        quality_metrics = {
            'all_tdoas': {k: v.tolist() for k, v in all_tdoas.items()},
            'all_weights': {k: v.tolist() for k, v in all_weights.items()},
            'n_arrays': len(received_signals),
            'cross_array_available': 'cross_array' in all_tdoas
        }

        return estimated_position, quality_metrics

    def multi_array_weighted_localization(self, mic_positions: Dict[str, np.ndarray], all_tdoas: Dict[str, np.ndarray], all_weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Enhanced weighted least squares localization using multiple arrays."""

        def weighted_objective_function(pos):
            x, y = pos
            source_pos = np.array([x, y])
            total_weighted_error = 0

            # Weighted error from internal array TDOAs
            for array_name in ['array_1', 'array_2']:
                if f'{array_name}_internal' in all_tdoas:
                    tdoas = all_tdoas[f'{array_name}_internal']
                    weights = all_weights[f'{array_name}_internal']
                    positions = mic_positions[array_name]

                    for i in range(len(tdoas)):
                        d0 = np.linalg.norm(source_pos - positions[0])
                        di = np.linalg.norm(source_pos - positions[i+1])
                        theoretical_tdoa = (di - d0) / self.speed_of_sound
                        error = (tdoas[i] - theoretical_tdoa) ** 2
                        # Apply weight based on correlation quality
                        total_weighted_error += weights[i] * error

            # Weighted error from cross-array TDOAs (stronger constraint)
            if 'cross_array' in all_tdoas:
                cross_tdoas = all_tdoas['cross_array']
                cross_weights = all_weights['cross_array']
                pos1 = mic_positions['array_1']
                pos2 = mic_positions['array_2']

                for i in range(len(cross_tdoas)):
                    d1 = np.linalg.norm(source_pos - pos1[i])
                    d2 = np.linalg.norm(source_pos - pos2[i])
                    theoretical_cross_tdoa = (d2 - d1) / self.speed_of_sound
                    error = (cross_tdoas[i] - theoretical_cross_tdoa) ** 2
                    # Weight cross-array TDOAs more heavily AND by correlation quality
                    total_weighted_error += 3.0 * cross_weights[i] * error

            return total_weighted_error

        # Initial guess - weighted centroid of array centers based on average weights
        array1_center = np.mean(mic_positions['array_1'], axis=0)
        array2_center = np.mean(mic_positions['array_2'], axis=0)

        # Calculate average weights for each array
        w1 = np.mean(all_weights.get('array_1_internal', [1.0]))
        w2 = np.mean(all_weights.get('array_2_internal', [1.0]))
        w_cross = np.mean(all_weights.get('cross_array', [1.0])) if 'cross_array' in all_weights else 1.0

        # Weighted initial guess
        total_weight = w1 + w2 + w_cross
        initial_guess = (w1 * array1_center + w2 * array2_center + w_cross * (array1_center + array2_center) / 2) / total_weight

        # Expanded search bounds for dual array system (dimension-specific with larger margins)
        all_positions = np.vstack([mic_positions[name] for name in mic_positions.keys()])
        x_extent = np.max(all_positions[:, 0]) - np.min(all_positions[:, 0])
        y_extent = np.max(all_positions[:, 1]) - np.min(all_positions[:, 1])

        # Use larger multiplier for Y bounds to accommodate realistic drone heights
        x_margin = x_extent * 1.5  # 1.5x array span
        y_margin = max(y_extent * 3.0, 300.0)  # At least 300m Y range for drone operations

        bounds = [
            (np.min(all_positions[:, 0]) - x_margin, np.max(all_positions[:, 0]) + x_margin),
            (np.min(all_positions[:, 1]) - y_margin, np.max(all_positions[:, 1]) + y_margin)
        ]

        # Solve weighted optimization problem
        from scipy.optimize import minimize
        result = minimize(weighted_objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            return result.x
        else:
            print(f"Warning: Weighted multi-array optimization failed. Using initial guess.")
            return initial_guess

    def multi_array_localization(self, mic_positions: Dict[str, np.ndarray], all_tdoas: Dict[str, np.ndarray]) -> np.ndarray:
        """Legacy method - kept for compatibility."""
        # Create dummy weights for backward compatibility
        all_weights = {}
        for key in all_tdoas.keys():
            all_weights[key] = np.ones_like(all_tdoas[key])

        return self.multi_array_weighted_localization(mic_positions, all_tdoas, all_weights)

    def calculate_theoretical_tdoas(self, source_pos: np.ndarray, mic_positions: np.ndarray) -> np.ndarray:
        """Calculate theoretical TDOAs for a given source position (for testing)."""
        tdoas = []

        # Distance from source to reference mic (mic 0)
        d0 = np.linalg.norm(source_pos - mic_positions[0])

        for i in range(1, len(mic_positions)):
            # Distance from source to mic i
            di = np.linalg.norm(source_pos - mic_positions[i])

            # TDOA (time difference)
            tdoa = (di - d0) / self.speed_of_sound
            tdoas.append(tdoa)

        return np.array(tdoas)