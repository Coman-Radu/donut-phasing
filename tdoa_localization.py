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

    def cross_correlation_tdoa(self, signal1: np.ndarray, signal2: np.ndarray,
                              analysis_mode: str = 'highest_peak') -> float:
        """Calculate TDOA between two signals using cross-correlation.

        Args:
            signal1, signal2: Input signals
            analysis_mode: 'highest_peak', 'first_peak', 'expected_range', 'analyze_all'
        """
        correlation = signal.correlate(signal2, signal1, mode='full')

        if analysis_mode == 'analyze_all':
            self._analyze_correlation_peaks(correlation, signal1, signal2)

        # Calculate time axis
        delay_samples_axis = np.arange(len(correlation)) - (len(signal1) - 1)
        time_delays = delay_samples_axis / self.sample_rate

        if analysis_mode == 'highest_peak':
            # Original method - highest absolute correlation
            peak_index = np.argmax(np.abs(correlation))

        elif analysis_mode == 'first_peak':
            # Find first significant peak (assumes direct path arrives first)
            abs_corr = np.abs(correlation)
            threshold = 0.3 * np.max(abs_corr)  # 30% of max correlation
            significant_peaks = abs_corr > threshold
            if np.any(significant_peaks):
                peak_index = np.argmax(significant_peaks)  # First True value
            else:
                peak_index = np.argmax(abs_corr)  # Fallback to highest

        elif analysis_mode == 'expected_range':
            # Only consider peaks within reasonable geometric constraints
            # Assume max reasonable TDOA is distance between mics / speed of sound
            max_reasonable_delay = 0.01  # 10ms max delay (about 3.4m at 343 m/s)
            reasonable_mask = np.abs(time_delays) <= max_reasonable_delay

            if np.any(reasonable_mask):
                masked_corr = np.abs(correlation) * reasonable_mask
                peak_index = np.argmax(masked_corr)
            else:
                peak_index = np.argmax(np.abs(correlation))  # Fallback
        else:
            # Default to highest peak
            peak_index = np.argmax(np.abs(correlation))

        # Calculate time delay
        delay_samples = delay_samples_axis[peak_index]
        time_delay = delay_samples / self.sample_rate

        return time_delay

    def _analyze_correlation_peaks(self, correlation: np.ndarray, signal1: np.ndarray, signal2: np.ndarray):
        """Analyze multiple peaks in correlation function."""
        from scipy.signal import find_peaks

        abs_corr = np.abs(correlation)
        delay_samples_axis = np.arange(len(correlation)) - (len(signal1) - 1)
        time_delays = delay_samples_axis / self.sample_rate

        # Find all significant peaks
        height_threshold = 0.2 * np.max(abs_corr)
        peaks, properties = find_peaks(abs_corr, height=height_threshold, distance=10)

        print(f"\nCorrelation Analysis:")
        print(f"Signal length: {len(signal1)} samples")
        print(f"Max correlation: {np.max(abs_corr):.4f}")
        print(f"Found {len(peaks)} significant peaks (>{height_threshold:.4f})")

        # Sort peaks by correlation strength
        peak_heights = abs_corr[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Descending order

        for i, peak_idx in enumerate(sorted_indices[:5]):  # Top 5 peaks
            peak_pos = peaks[peak_idx]
            delay_samples = delay_samples_axis[peak_pos]
            time_delay = delay_samples / self.sample_rate
            correlation_value = correlation[peak_pos]  # With sign

            print(f"  Peak {i+1}: TDOA = {time_delay:.6f}s, "
                  f"Correlation = {correlation_value:.4f}, "
                  f"Distance diff = {time_delay * self.speed_of_sound:.3f}m")

    def calculate_all_tdoas(self, received_signals: List[np.ndarray],
                           analysis_mode: str = 'highest_peak') -> np.ndarray:
        """Calculate TDOAs between all microphone pairs."""
        n_mics = len(received_signals)
        tdoas = []

        if analysis_mode == 'analyze_all':
            print(f"\n{'='*60}")
            print("CORRELATION PEAK ANALYSIS")
            print(f"{'='*60}")

        # Use first microphone as reference
        for i in range(1, n_mics):
            if analysis_mode == 'analyze_all':
                print(f"\nMicrophone pair: Reference (0) <-> Mic {i}")

            tdoa = self.cross_correlation_tdoa(received_signals[0], received_signals[i], analysis_mode)
            tdoas.append(tdoa)

        if analysis_mode == 'analyze_all':
            print(f"{'='*60}")

        return np.array(tdoas)

    def compare_peak_selection_methods(self, received_signals: List[np.ndarray],
                                     mic_positions: np.ndarray) -> Dict:
        """Compare different peak selection strategies."""
        methods = ['highest_peak', 'first_peak', 'expected_range']
        results = {}

        print(f"\n{'='*70}")
        print("COMPARING PEAK SELECTION STRATEGIES")
        print(f"{'='*70}")

        for method in methods:
            print(f"\n--- Method: {method.upper().replace('_', ' ')} ---")

            # Calculate TDOAs with this method
            tdoas = self.calculate_all_tdoas(received_signals, method)

            # Estimate position
            estimated_position = self.hyperbola_intersection_2d(mic_positions, tdoas)

            results[method] = {
                'tdoas': tdoas.tolist(),
                'position': estimated_position.tolist(),
                'tdoa_summary': {
                    'mean': float(np.mean(np.abs(tdoas))),
                    'std': float(np.std(tdoas)),
                    'range': [float(np.min(tdoas)), float(np.max(tdoas))]
                }
            }

            print(f"TDOAs: {[f'{t:.6f}' for t in tdoas]}")
            print(f"Position: [{estimated_position[0]:.3f}, {estimated_position[1]:.3f}]")

        return results

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

    def localize(self, received_signals: List[np.ndarray], mic_positions: np.ndarray,
                analysis_mode: str = 'highest_peak') -> Tuple[np.ndarray, Dict]:
        """Main localization function."""

        if analysis_mode == 'compare_all':
            # Compare all methods and return the comparison
            comparison_results = self.compare_peak_selection_methods(received_signals, mic_positions)

            # Use highest_peak as default for final result
            tdoas = self.calculate_all_tdoas(received_signals, 'highest_peak')
            estimated_position = self.hyperbola_intersection_2d(mic_positions, tdoas)

            quality_metrics = {
                'tdoas': tdoas.tolist(),
                'n_microphones': len(received_signals),
                'reference_mic_index': 0,
                'peak_comparison': comparison_results
            }
        else:
            # Calculate TDOAs with specified method
            tdoas = self.calculate_all_tdoas(received_signals, analysis_mode)

            # Estimate position using hyperbola intersection
            estimated_position = self.hyperbola_intersection_2d(mic_positions, tdoas)

            # Calculate some quality metrics
            quality_metrics = {
                'tdoas': tdoas.tolist(),
                'n_microphones': len(received_signals),
                'reference_mic_index': 0,
                'analysis_mode': analysis_mode
            }

        return estimated_position, quality_metrics

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