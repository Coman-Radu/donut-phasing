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
        mic_positions = self.microphone_array.get_all_positions()

        print(f"True drone position: {true_drone_position}")
        for array_name, positions in mic_positions.items():
            print(f"{array_name} positions: {positions}")

        print("Simulating signal reception...")
        # Simulate signal reception at all microphone arrays
        received_signals = self.microphone_array.receive_signal(
            drone_signal, true_drone_position, self.sample_rate,
            self.speed_of_sound, self.noise_level
        )

        print("Performing TDOA localization...")
        # Perform localization with multiple arrays
        estimated_position, quality_metrics = self.localizer.localize(
            received_signals, mic_positions
        )

        print(f"Estimated drone position: {estimated_position}")

        # Calculate error
        position_error = np.linalg.norm(estimated_position - true_drone_position)
        print(f"Position error: {position_error:.3f} meters")

        return true_drone_position, estimated_position, quality_metrics

    def plot_results(self, true_position: np.ndarray, estimated_position: np.ndarray, mic_range: float = 10.0):
        """Plot the localization results with dual microphone arrays."""
        all_mic_positions = self.microphone_array.get_all_positions()

        plt.figure(figsize=(12, 8))

        colors = {'array_1': 'blue', 'array_2': 'purple'}
        markers = {'array_1': '^', 'array_2': 's'}

        # Plot microphone ranges and positions for each array
        theta = np.linspace(0, 2*np.pi, 100)
        for array_name, mic_positions in all_mic_positions.items():
            color = colors.get(array_name, 'blue')
            marker = markers.get(array_name, '^')

            for i, pos in enumerate(mic_positions):
                range_x = pos[0] + mic_range * np.cos(theta)
                range_y = pos[1] + mic_range * np.sin(theta)
                plt.plot(range_x, range_y, '--', alpha=0.3, linewidth=1,
                        color=color, label=f'{array_name} Range' if i == 0 else "")

            # Plot microphones
            plt.scatter(mic_positions[:, 0], mic_positions[:, 1],
                       c=color, s=100, marker=marker, label=f'{array_name} Mics', zorder=5)

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

        # Add microphone labels for all arrays
        for array_name, mic_positions in all_mic_positions.items():
            for i, pos in enumerate(mic_positions):
                plt.annotate(f'{array_name[6:]}M{i}', (pos[0], pos[1]),
                            xytext=(8, 8), textcoords='offset points', fontsize=8, zorder=6)

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Drone Localization Results with Microphone Ranges')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

    def plot_tdoa_analysis(self, true_position: np.ndarray, estimated_position: np.ndarray, quality_metrics: Dict):
        """Plot TDOA analysis including hyperbolas and correlation data for multi-array system."""
        all_mic_positions = self.microphone_array.get_all_positions()
        all_tdoas = quality_metrics.get('all_tdoas', {})

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Hyperbolic curves and arrays
        ax1.set_title('Multi-Array TDOA Localization')

        # Create grid for hyperbola plotting (adjusted for larger scale)
        x_range = np.linspace(-50, 300, 200)
        y_range = np.linspace(-100, 250, 200)
        X, Y = np.meshgrid(x_range, y_range)

        # Plot arrays with different colors
        colors = {'array_1': 'blue', 'array_2': 'purple'}
        markers = {'array_1': '^', 'array_2': 's'}

        for array_name, mic_positions in all_mic_positions.items():
            color = colors.get(array_name, 'blue')
            marker = markers.get(array_name, '^')
            ax1.scatter(mic_positions[:, 0], mic_positions[:, 1],
                       c=color, s=100, marker=marker, label=f'{array_name}', zorder=5)

        # Plot cross-array baseline
        if len(all_mic_positions) >= 2:
            arrays = list(all_mic_positions.values())
            center1 = np.mean(arrays[0], axis=0)
            center2 = np.mean(arrays[1], axis=0)
            ax1.plot([center1[0], center2[0]], [center1[1], center2[1]],
                    'k-', linewidth=2, alpha=0.5, label='Cross-Array Baseline')

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

        # Plot 2: Multi-Array TDOA comparison
        ax2.set_title('Multi-Array TDOA Analysis')

        # Combine all TDOA measurements
        all_labels = []
        all_measured = []

        for tdoa_type, values in all_tdoas.items():
            if 'internal' in tdoa_type:
                array_name = tdoa_type.split('_')[0]
                for i, val in enumerate(values):
                    all_labels.append(f'{array_name}_{i+1}')
                    all_measured.append(val)
            elif tdoa_type == 'cross_array':
                for i, val in enumerate(values):
                    all_labels.append(f'Cross_{i}')
                    all_measured.append(val)

        if all_measured:
            x_pos = np.arange(len(all_labels))
            ax2.bar(x_pos, all_measured, alpha=0.7,
                   color=['blue' if 'array1' in label else 'purple' if 'array2' in label else 'red' for label in all_labels])
            ax2.set_xlabel('TDOA Measurements')
            ax2.set_ylabel('TDOA (seconds)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(all_labels, rotation=45)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Distance visualization for all arrays
        ax3.set_title('Distance from Drone to All Microphones')

        all_distances_true = []
        all_distances_est = []
        all_mic_labels = []

        for array_name, mic_positions in all_mic_positions.items():
            for i, mic_pos in enumerate(mic_positions):
                dist_true = np.linalg.norm(true_position - mic_pos)
                dist_est = np.linalg.norm(estimated_position - mic_pos)
                all_distances_true.append(dist_true)
                all_distances_est.append(dist_est)
                all_mic_labels.append(f'{array_name[6:]}_M{i}')

        if all_distances_true:
            x_pos = np.arange(len(all_mic_labels))
            width = 0.35
            ax3.bar(x_pos - width/2, all_distances_true, width, label='True Distances', alpha=0.7)
            ax3.bar(x_pos + width/2, all_distances_est, width, label='Estimated Distances', alpha=0.7)

            ax3.set_xlabel('Microphones')
            ax3.set_ylabel('Distance (m)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(all_mic_labels, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Cross-Array TDOA Analysis
        ax4.set_title('Cross-Array TDOA Values')

        if 'cross_array' in all_tdoas:
            cross_tdoas = all_tdoas['cross_array']
            cross_labels = [f'Cross_Mic_{i}' for i in range(len(cross_tdoas))]
            x_pos = np.arange(len(cross_labels))

            ax4.bar(x_pos, cross_tdoas, alpha=0.7, color='red')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Cross-Array Microphone Pairs')
            ax4.set_ylabel('TDOA (seconds)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cross_labels)
            ax4.grid(True, alpha=0.3)

            # Add statistics text
            mean_cross = np.mean(np.abs(cross_tdoas))
            max_cross = np.max(np.abs(cross_tdoas))
            ax4.text(0.02, 0.95, f'Mean |Cross-TDOA|: {mean_cross:.6f}s\nMax |Cross-TDOA|: {max_cross:.6f}s\nBaseline: ~200m',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Cross-Array TDOAs Available',
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_wave_arrivals(self, received_signals: List[np.ndarray], true_position: np.ndarray, array_name: str = 'array_1'):
        """Plot the wave arrivals at each microphone with time delays."""
        mic_positions = self.microphone_array.get_positions(array_name)

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

            ax.set_ylabel(f'{array_name} Mic {i}\nAmplitude')
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
        plt.suptitle(f'Wave Arrivals at {array_name}\nDrone at ({true_position[0]:.2f}, {true_position[1]:.2f})')
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
        all_positions = self.microphone_array.get_all_positions()
        total_mics = sum(len(positions) for positions in all_positions.values())

        return {
            'microphone_count': total_mics,
            'n_arrays': len(all_positions),
            'microphone_positions': {name: pos.tolist() for name, pos in all_positions.items()},
            'drone_position': self.drone_signal_generator.get_position().tolist(),
            'signal_type': self.drone_signal_generator.signal_type,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'speed_of_sound': self.speed_of_sound,
            'noise_level': self.noise_level
        }