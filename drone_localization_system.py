import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from microphone_array import MicrophoneArray
from drone_signal import DroneSignalGenerator
from tdoa_localization import TDOALocalizer


class DroneLocalizationSystem:
    def __init__(self, config_path: str = 'config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.microphone_array = MicrophoneArray(config_dict=self.config)
        self.drone_signal_generator = DroneSignalGenerator(config_dict=self.config)
        self.localizer = TDOALocalizer(config_dict=self.config)

        # Simulation parameters
        self.sample_rate = self.config['simulation']['sample_rate']
        self.duration = self.config['simulation']['duration']
        self.noise_level = self.config['simulation']['noise_level']
        self.speed_of_sound = self.config['simulation']['speed_of_sound']

    def run_simulation(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run complete drone localization simulation."""

        print("Generating drone signal...")
        # Generate drone signal
        drone_signal = self.drone_signal_generator.generate_signal(
            self.duration, self.sample_rate
        )

        # Get drone and microphone positions
        true_drone_position = self.drone_signal_generator.get_position()
        mic_positions = self.microphone_array.get_positions()

        print(f"True drone position: {true_drone_position}")
        print(f"Microphone positions: {mic_positions}")

        print("Simulating signal reception...")
        # Simulate signal reception at microphones
        received_signals = self.microphone_array.receive_signal(
            drone_signal, true_drone_position, self.sample_rate,
            self.speed_of_sound, self.noise_level
        )

        print("Performing TDOA localization...")
        # Perform localization
        estimated_position, quality_metrics = self.localizer.localize(
            received_signals, mic_positions
        )

        print(f"Estimated drone position: {estimated_position}")

        # Calculate error
        position_error = np.linalg.norm(estimated_position - true_drone_position)
        print(f"Position error: {position_error:.3f} meters")

        return true_drone_position, estimated_position, quality_metrics

    def plot_results(self, true_position: np.ndarray, estimated_position: np.ndarray, mic_range: float = 10.0):
        """Plot the localization results with microphone ranges."""
        mic_positions = self.microphone_array.get_positions()

        plt.figure(figsize=(10, 8))

        # Plot microphone ranges with broken circular lines
        theta = np.linspace(0, 2*np.pi, 100)
        for i, pos in enumerate(mic_positions):
            range_x = pos[0] + mic_range * np.cos(theta)
            range_y = pos[1] + mic_range * np.sin(theta)
            plt.plot(range_x, range_y, '--', alpha=0.4, linewidth=1,
                    color='lightblue', label='Mic Range' if i == 0 else "")

        # Plot microphones
        plt.scatter(mic_positions[:, 0], mic_positions[:, 1],
                   c='blue', s=100, marker='^', label='Microphones', zorder=5)

        # Plot true drone position with donut shape (circle with facecolor='none')
        plt.scatter(true_position[0], true_position[1],
                   c='red', s=150, marker='o', facecolors='none',
                   edgecolors='red', linewidth=2, label='True Drone Position', zorder=5)

        # Plot estimated drone position
        plt.scatter(estimated_position[0], estimated_position[1],
                   c='green', s=150, marker='x', linewidth=3, label='Estimated Position', zorder=5)

        # Draw error line
        plt.plot([true_position[0], estimated_position[0]],
                [true_position[1], estimated_position[1]],
                'k--', alpha=0.7, linewidth=2, label='Position Error', zorder=4)

        # Add microphone labels
        for i, pos in enumerate(mic_positions):
            plt.annotate(f'Mic {i}', (pos[0], pos[1]),
                        xytext=(8, 8), textcoords='offset points', fontsize=9, zorder=6)

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Drone Localization Results with Microphone Ranges')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

    def update_drone_position(self, new_position: List[float]):
        """Update drone position for multiple runs."""
        self.drone_signal_generator.update_position(np.array(new_position))
        self.config['drone']['position'] = new_position

    def update_config(self, new_config: Dict):
        """Update configuration parameters."""
        self.config.update(new_config)
        # Recreate components with new config
        self.microphone_array = MicrophoneArray(config_dict=self.config)
        self.drone_signal_generator = DroneSignalGenerator(config_dict=self.config)
        self.localizer = TDOALocalizer(config_dict=self.config)

    def save_config(self, filename: str = 'config_modified.json'):
        """Save current configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {filename}")

    def get_system_info(self) -> Dict:
        """Get information about the current system setup."""
        return {
            'microphone_count': len(self.microphone_array.microphones),
            'microphone_positions': self.microphone_array.get_positions().tolist(),
            'drone_position': self.drone_signal_generator.get_position().tolist(),
            'signal_type': self.drone_signal_generator.signal_type,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'speed_of_sound': self.speed_of_sound,
            'noise_level': self.noise_level
        }