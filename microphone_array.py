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

        self.arrays = {}
        mic_config = config['microphones']

        # Multi-array format only
        for array_name in ['array_1', 'array_2']:
            if array_name in mic_config:
                self.arrays[array_name] = []
                for pos in mic_config[array_name]['positions']:
                    mic = Microphone(
                        position=pos,
                        sensitivity=mic_config['sensitivity'],
                        frequency_response=mic_config['frequency_response'],
                        dynamic_range=mic_config['dynamic_range'],
                        directionality=mic_config['directionality'],
                        audio_transparency=mic_config['audio_transparency'],
                        gps_sync=mic_config['gps_sync']
                    )
                    self.arrays[array_name].append(mic)

    def get_positions(self, array_name: str = 'array_1') -> np.ndarray:
        return np.array([mic.position for mic in self.arrays[array_name]])

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        return {array_name: self.get_positions(array_name) for array_name in self.arrays}

    def receive_signal(self, source_signal: np.ndarray, source_position: np.ndarray,
                      sample_rate: int, speed_of_sound: float, noise_level: float = 0.0) -> Dict[str, List[np.ndarray]]:
        all_received_signals = {}

        # Find maximum delay to ensure all signals have same length
        max_sample_delay = 0
        delays = {}

        for array_name, mics in self.arrays.items():
            delays[array_name] = []
            for mic in mics:
                distance = np.linalg.norm(source_position - mic.position)
                time_delay = distance / speed_of_sound
                sample_delay = int(time_delay * sample_rate)
                delays[array_name].append(sample_delay)
                max_sample_delay = max(max_sample_delay, sample_delay)

        for array_name, mics in self.arrays.items():
            received_signals = []
            for i, mic in enumerate(mics):
                sample_delay = delays[array_name][i]

                # Create signal with consistent length by padding to max_delay + original length
                total_length = len(source_signal) + max_sample_delay
                delayed_signal = np.zeros(total_length)

                # Place the source signal at the correct delay position
                delayed_signal[sample_delay:sample_delay + len(source_signal)] = source_signal

                # Truncate to original signal length, keeping the end portion
                # This preserves the relative delays between microphones
                processed_signal = delayed_signal[-len(source_signal):]

                # Calculate angle from microphone to source
                direction_vector = source_position - mic.position
                source_angle = np.arctan2(direction_vector[1], direction_vector[0])

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

            all_received_signals[array_name] = received_signals

        return all_received_signals