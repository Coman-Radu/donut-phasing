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

    def cross_correlation_tdoa(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate TDOA between two signals using cross-correlation."""
        correlation = signal.correlate(signal2, signal1, mode='full')

        # Find the peak correlation
        peak_index = np.argmax(np.abs(correlation))

        # Calculate time delay in samples
        delay_samples = peak_index - (len(signal1) - 1)

        # Convert to time delay in seconds
        time_delay = delay_samples / self.sample_rate

        return time_delay

    def calculate_all_tdoas(self, received_signals: List[np.ndarray]) -> np.ndarray:
        """Calculate TDOAs between all microphone pairs using cross-correlation."""
        n_mics = len(received_signals)
        tdoas = []

        # Use first microphone as reference
        for i in range(1, n_mics):
            # Apply windowing to reduce edge effects for pulsed signals
            window = np.hanning(len(received_signals[0]))
            sig1_windowed = received_signals[0] * window
            sig2_windowed = received_signals[i] * window

            tdoa = self.cross_correlation_tdoa(sig1_windowed, sig2_windowed)
            tdoas.append(tdoa)

        return np.array(tdoas)

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

    def localize(self, received_signals: List[np.ndarray], mic_positions: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Main localization function."""

        # Calculate TDOAs
        tdoas = self.calculate_all_tdoas(received_signals)

        # Estimate position using hyperbola intersection
        estimated_position = self.hyperbola_intersection_2d(mic_positions, tdoas)

        # Calculate some quality metrics
        quality_metrics = {
            'tdoas': tdoas.tolist(),
            'n_microphones': len(received_signals),
            'reference_mic_index': 0
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