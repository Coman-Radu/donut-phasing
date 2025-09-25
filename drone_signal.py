import numpy as np
import json
from typing import Dict, Any
import wave
import os


class DroneSignalGenerator:
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = config_dict

        self.drone_config = config['drone']
        self.signal_type = self.drone_config['signal_type']
        self.signal_config = self.drone_config['signal_config']
        self.position = np.array(self.drone_config['position'])

    def generate_signal(self, duration: float, sample_rate: int) -> np.ndarray:
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        if self.signal_type == 'sine':
            return self._generate_sine_wave(t)
        elif self.signal_type == 'complex':
            return self._generate_complex_wave(t)
        elif self.signal_type == 'wav_file':
            return self._load_wav_file(duration, sample_rate)
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")

    def _generate_sine_wave(self, t: np.ndarray) -> np.ndarray:
        config = self.signal_config['sine']
        frequency = config['frequency']
        amplitude = config['amplitude']

        return amplitude * np.sin(2 * np.pi * frequency * t)

    def _generate_complex_wave(self, t: np.ndarray) -> np.ndarray:
        config = self.signal_config['complex']
        fundamental_freq = config['fundamental_frequency']
        harmonics = config['harmonics']
        harmonic_amplitudes = config['harmonic_amplitudes']
        phases = config['phases']

        signal = np.zeros_like(t)

        for harmonic, amplitude, phase in zip(harmonics, harmonic_amplitudes, phases):
            frequency = fundamental_freq * harmonic
            signal += amplitude * np.sin(2 * np.pi * frequency * t + phase)

        return signal

    def _load_wav_file(self, duration: float, sample_rate: int) -> np.ndarray:
        config = self.signal_config['wav_file']
        wav_path = config['path']
        loop = config.get('loop', True)

        if not os.path.exists(wav_path):
            print(f"Warning: WAV file {wav_path} not found. Generating sine wave instead.")
            return self._generate_fallback_sine(duration, sample_rate)

        try:
            with wave.open(wav_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize

                original_sample_rate = wav_file.getframerate()

                # Resample if necessary
                if original_sample_rate != sample_rate:
                    from scipy import signal as scipy_signal
                    num_samples = int(len(audio_data) * sample_rate / original_sample_rate)
                    audio_data = scipy_signal.resample(audio_data, num_samples)

                # Loop or trim to desired duration
                desired_length = int(duration * sample_rate)

                if loop and len(audio_data) < desired_length:
                    repeats = int(np.ceil(desired_length / len(audio_data)))
                    audio_data = np.tile(audio_data, repeats)

                return audio_data[:desired_length]

        except Exception as e:
            print(f"Error loading WAV file: {e}. Generating sine wave instead.")
            return self._generate_fallback_sine(duration, sample_rate)

    def _generate_fallback_sine(self, duration: float, sample_rate: int) -> np.ndarray:
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        return np.sin(2 * np.pi * 440 * t)  # 440 Hz fallback

    def get_position(self) -> np.ndarray:
        return self.position

    def update_position(self, new_position: np.ndarray):
        self.position = new_position