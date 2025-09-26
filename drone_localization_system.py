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

    def plot_tdoa_analysis(self, true_position: np.ndarray, estimated_position: np.ndarray, quality_metrics: Dict):
        """Plot TDOA analysis including hyperbolas and correlation data."""
        mic_positions = self.microphone_array.get_positions()
        tdoas = np.array(quality_metrics['tdoas'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Hyperbolic curves from TDOA measurements
        ax1.set_title('TDOA Hyperbolas and Localization')

        # Create grid for hyperbola plotting (adjusted for larger scale)
        x_range = np.linspace(-50, 300, 200)
        y_range = np.linspace(-100, 250, 200)
        X, Y = np.meshgrid(x_range, y_range)

        # Plot hyperbolas for each TDOA measurement
        colors = ['red', 'green', 'blue']
        for i, tdoa in enumerate(tdoas):
            if i < len(colors):
                # Calculate theoretical hyperbola
                mic0 = mic_positions[0]
                mic_i = mic_positions[i+1]

                # Distance differences
                d0 = np.sqrt((X - mic0[0])**2 + (Y - mic0[1])**2)
                di = np.sqrt((X - mic_i[0])**2 + (Y - mic_i[1])**2)

                # TDOA hyperbola equation: di - d0 = c * tdoa
                hyperbola = di - d0 - self.speed_of_sound * tdoa

                # Plot contour where hyperbola = 0
                ax1.contour(X, Y, hyperbola, levels=[0], colors=[colors[i]],
                           linewidths=2, alpha=0.7,
                           label=f'TDOA {i+1}: {tdoa:.4f}s')

        # Plot microphones and positions
        ax1.scatter(mic_positions[:, 0], mic_positions[:, 1],
                   c='blue', s=100, marker='^', label='Microphones', zorder=5)
        ax1.scatter(true_position[0], true_position[1],
                   c='red', s=150, marker='o', facecolors='none',
                   edgecolors='red', linewidth=2, label='True Position', zorder=5)
        ax1.scatter(estimated_position[0], estimated_position[1],
                   c='green', s=150, marker='x', linewidth=3, label='Estimated', zorder=5)

        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot 2: TDOA values comparison
        ax2.set_title('TDOA Measurements vs Theoretical')
        theoretical_tdoas = self.localizer.calculate_theoretical_tdoas(true_position, mic_positions)

        mic_pairs = [f'Mic 0-{i+1}' for i in range(len(tdoas))]
        x_pos = np.arange(len(mic_pairs))

        width = 0.35
        ax2.bar(x_pos - width/2, tdoas, width, label='Measured TDOA', alpha=0.7)
        ax2.bar(x_pos + width/2, theoretical_tdoas, width, label='Theoretical TDOA', alpha=0.7)

        ax2.set_xlabel('Microphone Pairs')
        ax2.set_ylabel('TDOA (seconds)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(mic_pairs)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distance visualization
        ax3.set_title('Distance from Drone to Each Microphone')
        distances_true = [np.linalg.norm(true_position - mic) for mic in mic_positions]
        distances_est = [np.linalg.norm(estimated_position - mic) for mic in mic_positions]

        mic_labels = [f'Mic {i}' for i in range(len(mic_positions))]
        x_pos = np.arange(len(mic_labels))

        ax3.bar(x_pos - width/2, distances_true, width, label='True Distances', alpha=0.7)
        ax3.bar(x_pos + width/2, distances_est, width, label='Estimated Distances', alpha=0.7)

        ax3.set_xlabel('Microphones')
        ax3.set_ylabel('Distance (m)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(mic_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Error analysis
        ax4.set_title('TDOA Error Analysis')
        tdoa_errors = tdoas - theoretical_tdoas

        ax4.bar(mic_pairs, tdoa_errors, alpha=0.7, color='red')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Microphone Pairs')
        ax4.set_ylabel('TDOA Error (seconds)')
        ax4.grid(True, alpha=0.3)

        # Add error statistics text
        mean_error = np.mean(np.abs(tdoa_errors))
        max_error = np.max(np.abs(tdoa_errors))
        ax4.text(0.02, 0.95, f'Mean |Error|: {mean_error:.6f}s\nMax |Error|: {max_error:.6f}s',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_wave_arrivals(self, received_signals: List[np.ndarray], true_position: np.ndarray):
        """Plot the wave arrivals at each microphone with time delays."""
        mic_positions = self.microphone_array.get_positions()

        # Calculate theoretical arrival times
        distances = [np.linalg.norm(true_position - mic_pos) for mic_pos in mic_positions]
        arrival_times = [d / self.speed_of_sound for d in distances]

        # Create time axis
        duration = len(received_signals[0]) / self.sample_rate
        time_axis = np.linspace(0, duration, len(received_signals[0]))

        fig, axes = plt.subplots(len(received_signals), 1, figsize=(12, 8))
        if len(received_signals) == 1:
            axes = [axes]

        # Find global min/max for consistent y-axis scaling
        global_min = min(np.min(sig) for sig in received_signals)
        global_max = max(np.max(sig) for sig in received_signals)

        for i, (signal, ax) in enumerate(zip(received_signals, axes)):
            # Plot the received signal
            ax.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.8)

            # Mark theoretical arrival time
            ax.axvline(x=arrival_times[i], color='red', linestyle='--',
                      linewidth=2, alpha=0.7, label=f'Expected arrival: {arrival_times[i]:.4f}s')

            # Mark signal start (first significant peak)
            signal_energy = signal ** 2
            threshold = 0.1 * np.max(signal_energy)
            signal_start_idx = np.where(signal_energy > threshold)[0]

            if len(signal_start_idx) > 0:
                actual_arrival_time = time_axis[signal_start_idx[0]]
                ax.axvline(x=actual_arrival_time, color='green', linestyle='-',
                          linewidth=2, alpha=0.7, label=f'Detected arrival: {actual_arrival_time:.4f}s')

                # Calculate time difference
                time_diff = actual_arrival_time - arrival_times[i]
                ax.text(0.02, 0.95, f'Time diff: {time_diff:.4f}s\nDistance: {distances[i]:.2f}m',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_ylabel(f'Mic {i}\nAmplitude')
            ax.set_ylim(global_min * 1.1, global_max * 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Highlight pulse regions
            if hasattr(self.drone_signal_generator.signal_config.get('sine', {}), '__contains__'):
                if 'pulse_interval' in self.drone_signal_generator.signal_config['sine']:
                    pulse_interval = self.drone_signal_generator.signal_config['sine']['pulse_interval']
                    pulse_width = self.drone_signal_generator.signal_config['sine']['pulse_width']

                    current_time = arrival_times[i]
                    while current_time < duration:
                        ax.axvspan(current_time, current_time + pulse_width,
                                  alpha=0.2, color='yellow', label='Expected pulse' if current_time == arrival_times[i] else '')
                        current_time += pulse_interval

        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(f'Wave Arrivals at Microphones\nDrone at ({true_position[0]:.2f}, {true_position[1]:.2f})')
        plt.tight_layout()
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