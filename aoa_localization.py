import numpy as np
from scipy import signal
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import json


class AOALocalizer:
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = config_dict

        self.speed_of_sound = config['simulation']['speed_of_sound']
        self.sample_rate = config['simulation']['sample_rate']

    def calculate_phase_difference(self, signal1: np.ndarray, signal2: np.ndarray,
                                 frequency: float) -> float:
        """Calculate phase difference between two signals at a specific frequency."""
        # Apply windowing to reduce spectral leakage
        window = np.hanning(len(signal1))
        windowed_signal1 = signal1 * window
        windowed_signal2 = signal2 * window

        # Compute FFT
        fft1 = np.fft.fft(windowed_signal1)
        fft2 = np.fft.fft(windowed_signal2)

        # Find frequency bin closest to target frequency
        freqs = np.fft.fftfreq(len(signal1), 1/self.sample_rate)
        freq_idx = np.argmin(np.abs(freqs - frequency))

        # Calculate phase difference
        phase1 = np.angle(fft1[freq_idx])
        phase2 = np.angle(fft2[freq_idx])

        phase_diff = phase2 - phase1

        # Wrap phase difference to [-π, π]
        phase_diff = np.angle(np.exp(1j * phase_diff))

        return phase_diff

    def calculate_all_phase_differences(self, received_signals: List[np.ndarray],
                                      frequency: float) -> np.ndarray:
        """Calculate phase differences between all microphone pairs."""
        n_mics = len(received_signals)
        phase_diffs = []

        # Use first microphone as reference
        for i in range(1, n_mics):
            phase_diff = self.calculate_phase_difference(
                received_signals[0], received_signals[i], frequency)
            phase_diffs.append(phase_diff)

        return np.array(phase_diffs)

    def phase_diff_to_angle(self, phase_diff: float, frequency: float,
                           mic_separation: float) -> float:
        """Convert phase difference to angle of arrival."""
        wavelength = self.speed_of_sound / frequency

        # Phase difference to path difference
        path_diff = (phase_diff * wavelength) / (2 * np.pi)

        # Path difference to angle (assuming far-field)
        # sin(θ) = path_difference / mic_separation
        sin_theta = path_diff / mic_separation

        # Handle cases where |sin_theta| > 1 due to noise/aliasing
        sin_theta = np.clip(sin_theta, -1.0, 1.0)

        angle = np.arcsin(sin_theta)
        return angle

    def localize(self, received_signals: List[np.ndarray], mic_positions: np.ndarray,
                frequency: float = None) -> Tuple[np.ndarray, Dict]:
        """Main AOA localization function."""

        # If no frequency specified, estimate dominant frequency
        if frequency is None:
            frequency = self._estimate_dominant_frequency(received_signals[0])

        # Calculate phase differences
        phase_diffs = self.calculate_all_phase_differences(received_signals, frequency)

        # Calculate angles from phase differences
        angles = []
        ref_pos = mic_positions[0]

        for i, phase_diff in enumerate(phase_diffs):
            mic_pos = mic_positions[i + 1]
            mic_separation = np.linalg.norm(mic_pos - ref_pos)
            angle = self.phase_diff_to_angle(phase_diff, frequency, mic_separation)
            angles.append(angle)

        angles = np.array(angles)

        # Estimate position using triangulation from angles
        estimated_position = self._triangulate_from_angles(mic_positions, angles)

        quality_metrics = {
            'frequency_used': frequency,
            'phase_differences': phase_diffs.tolist(),
            'angles': angles.tolist(),
            'n_microphones': len(received_signals)
        }

        return estimated_position, quality_metrics

    def _estimate_dominant_frequency(self, signal: np.ndarray) -> float:
        """Estimate the dominant frequency in the signal."""
        window = np.hanning(len(signal))
        windowed_signal = signal * window

        fft = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)

        # Only consider positive frequencies
        positive_freq_idx = freqs > 0
        magnitude = np.abs(fft[positive_freq_idx])
        positive_freqs = freqs[positive_freq_idx]

        # Find frequency with maximum magnitude
        max_idx = np.argmax(magnitude)
        dominant_frequency = positive_freqs[max_idx]

        return dominant_frequency

    def _triangulate_from_angles(self, mic_positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Triangulate source position from angles of arrival."""
        # Use optimization to find best fit position
        def objective_function(pos):
            x, y = pos
            source_pos = np.array([x, y])

            error = 0
            ref_pos = mic_positions[0]

            for i, measured_angle in enumerate(angles):
                mic_pos = mic_positions[i + 1]

                # Vector from reference mic to source
                ref_to_source = source_pos - ref_pos
                # Vector from reference mic to current mic
                ref_to_mic = mic_pos - ref_pos

                # Calculate expected angle based on geometry
                expected_angle = np.arctan2(
                    np.cross(ref_to_source, ref_to_mic),
                    np.dot(ref_to_source, ref_to_mic)
                )

                # Calculate error
                angle_error = np.abs(np.angle(np.exp(1j * (expected_angle - measured_angle))))
                error += angle_error ** 2

            return error

        # Initial guess - centroid of microphones
        initial_guess = np.mean(mic_positions, axis=0)

        # Optimization bounds
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
            print(f"Warning: AOA optimization failed. Using initial guess.")
            return initial_guess