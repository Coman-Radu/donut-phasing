import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Any
import json


class Microphone:
    def __init__(self, position: Tuple[float, float], sensitivity: float,
                 frequency_response: Dict, dynamic_range: float,
                 directionality: Dict, audio_transparency: float, gps_sync: Dict):
        self.position = np.array(position)
        self.sensitivity = sensitivity  # dB
        self.frequency_response = frequency_response
        self.dynamic_range = dynamic_range  # dB
        self.directionality = directionality
        self.audio_transparency = audio_transparency
        self.gps_sync = gps_sync

    def apply_frequency_response(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        low_cutoff = self.frequency_response['low_cutoff']
        high_cutoff = self.frequency_response['high_cutoff']

        nyquist = sample_rate / 2
        low = low_cutoff / nyquist
        high = min(high_cutoff / nyquist, 0.99)

        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, signal_data)

    def apply_directionality(self, signal_data: np.ndarray, source_angle: float) -> np.ndarray:
        if self.directionality['type'] == 'omnidirectional':
            return signal_data
        elif self.directionality['type'] == 'cardioid':
            pattern_coeff = self.directionality.get('pattern_coefficient', 0.5)
            gain = pattern_coeff + (1 - pattern_coeff) * np.cos(source_angle)
            return signal_data * max(0, gain)
        return signal_data

    def apply_sensitivity(self, signal_data: np.ndarray) -> np.ndarray:
        sensitivity_linear = 10 ** (self.sensitivity / 20)
        return signal_data * sensitivity_linear

    def apply_transparency(self, signal_data: np.ndarray) -> np.ndarray:
        return signal_data * self.audio_transparency

    def apply_gps_sync_error(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.gps_sync['enabled']:
            return signal_data

        # Simulate GPS synchronization errors
        clock_drift = self.gps_sync['clock_drift']
        sync_accuracy = self.gps_sync['sync_accuracy']

        # Add small time drift (clock drift over duration)
        duration = len(signal_data) / sample_rate
        total_drift = clock_drift * duration

        # Add random sync error
        sync_error = np.random.normal(0, sync_accuracy)

        # Combined timing error in seconds
        timing_error = total_drift + sync_error

        # Convert to sample offset
        sample_offset = int(timing_error * sample_rate)

        if sample_offset != 0:
            # Apply timing offset
            if sample_offset > 0:
                # Delay signal
                delayed_signal = np.zeros(len(signal_data) + sample_offset)
                delayed_signal[sample_offset:] = signal_data
                return delayed_signal[:len(signal_data)]
            else:
                # Advance signal
                return np.pad(signal_data[-sample_offset:], (0, -sample_offset), 'constant')

        return signal_data


class MicrophoneArray:
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = config_dict

        self.microphones = []
        mic_config = config['microphones']

        for pos in mic_config['positions']:
            mic = Microphone(
                position=pos,
                sensitivity=mic_config['sensitivity'],
                frequency_response=mic_config['frequency_response'],
                dynamic_range=mic_config['dynamic_range'],
                directionality=mic_config['directionality'],
                audio_transparency=mic_config['audio_transparency'],
                gps_sync=mic_config['gps_sync']
            )
            self.microphones.append(mic)

    def get_positions(self) -> np.ndarray:
        return np.array([mic.position for mic in self.microphones])

    def receive_signal(self, source_signal: np.ndarray, source_position: np.ndarray,
                      sample_rate: int, speed_of_sound: float, noise_level: float = 0.0) -> List[np.ndarray]:
        received_signals = []

        for mic in self.microphones:
            # Calculate distance and time delay
            distance = np.linalg.norm(source_position - mic.position)
            time_delay = distance / speed_of_sound
            sample_delay = int(time_delay * sample_rate)

            # Apply time delay by zero-padding
            delayed_signal = np.zeros(len(source_signal) + sample_delay)
            delayed_signal[sample_delay:] = source_signal

            # Calculate angle from microphone to source
            direction_vector = source_position - mic.position
            source_angle = np.arctan2(direction_vector[1], direction_vector[0])

            # Apply microphone characteristics
            processed_signal = delayed_signal[:len(source_signal)]
            processed_signal = mic.apply_sensitivity(processed_signal)
            processed_signal = mic.apply_directionality(processed_signal, source_angle)
            processed_signal = mic.apply_frequency_response(processed_signal, sample_rate)
            processed_signal = mic.apply_transparency(processed_signal)

            # Apply GPS synchronization errors
            processed_signal = mic.apply_gps_sync_error(processed_signal, sample_rate)

            # Add noise
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, len(processed_signal))
                processed_signal += noise

            received_signals.append(processed_signal)

        return received_signals