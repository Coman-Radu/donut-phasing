#!/usr/bin/env python3
"""
Real-time TDOA response test over 20 seconds using WAV file
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drone_localization_system import DroneLocalizationSystem
import time

class RealTimeTDOATest:
    def __init__(self, config_path='config.json'):
        self.system = DroneLocalizationSystem(config_path)

        # Time window parameters
        self.window_duration = 1.0  # 1 second analysis windows
        self.total_duration = 20.0  # 20 second total test
        self.overlap = 0.5  # 50% overlap between windows

        # Storage for results
        self.time_points = []
        self.tdoa_history = []
        self.weight_history = []
        self.error_history = []
        self.position_history = []

    def run_realtime_analysis(self):
        """Run real-time TDOA analysis over 20 seconds."""
        print("Generating 20-second drone audio signal...")

        # Generate full 20-second signal
        drone_signal = self.system.drone_signal_generator.generate_signal(
            self.total_duration, self.system.sample_rate
        )

        true_drone_position = self.system.drone_signal_generator.get_position()
        all_mic_positions = self.system.microphone_array.get_all_positions()

        print(f"True drone position: {true_drone_position}")
        print(f"Total signal length: {len(drone_signal)/self.system.sample_rate:.1f} seconds")

        # Generate received signals for all microphones
        print("Simulating signal reception at all microphones...")
        all_received_signals = self.system.microphone_array.receive_signal(
            drone_signal, true_drone_position, self.system.sample_rate,
            self.system.speed_of_sound, self.system.noise_level
        )

        # Analyze in time windows
        samples_per_window = int(self.window_duration * self.system.sample_rate)
        step_size = int(samples_per_window * (1 - self.overlap))

        window_count = 0
        start_sample = 0

        while start_sample + samples_per_window <= len(drone_signal):
            end_sample = start_sample + samples_per_window
            current_time = start_sample / self.system.sample_rate

            print(f"Analyzing window {window_count+1} (t={current_time:.1f}-{current_time+self.window_duration:.1f}s)")

            # Extract windowed signals for each array
            windowed_signals = {}
            for array_name, signals in all_received_signals.items():
                windowed_signals[array_name] = [
                    signal[start_sample:end_sample] for signal in signals
                ]

            # Perform TDOA localization on this window
            estimated_position, metrics = self.system.localizer.localize(
                windowed_signals, all_mic_positions
            )

            # Calculate error
            error = np.linalg.norm(estimated_position - true_drone_position)

            # Store results
            self.time_points.append(current_time + self.window_duration/2)  # Center of window
            self.position_history.append(estimated_position.copy())
            self.error_history.append(error)

            # Store TDOA and weight data
            tdoa_snapshot = {}
            weight_snapshot = {}

            if 'all_tdoas' in metrics:
                for key, values in metrics['all_tdoas'].items():
                    tdoa_snapshot[key] = values.copy()

            if 'all_weights' in metrics:
                for key, values in metrics['all_weights'].items():
                    weight_snapshot[key] = values.copy()

            self.tdoa_history.append(tdoa_snapshot)
            self.weight_history.append(weight_snapshot)

            print(f"  Position: ({estimated_position[0]:.1f}, {estimated_position[1]:.1f}), Error: {error:.1f}m")

            # Move to next window
            start_sample += step_size
            window_count += 1

        print(f"Analysis complete! Processed {window_count} time windows")

    def plot_realtime_results(self):
        """Create comprehensive real-time TDOA visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: TDOA values over time
        ax1.set_title('TDOA Values Over Time (20 seconds)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('TDOA (seconds)')

        # Plot different TDOA types with different colors
        colors = {'array_1_internal': 'blue', 'array_2_internal': 'green', 'cross_array': 'red'}

        for tdoa_type in colors.keys():
            values_over_time = []
            for tdoa_snapshot in self.tdoa_history:
                if tdoa_type in tdoa_snapshot:
                    # Take average of all TDOAs for this type
                    avg_tdoa = np.mean(tdoa_snapshot[tdoa_type])
                    values_over_time.append(avg_tdoa)
                else:
                    values_over_time.append(np.nan)

            ax1.plot(self.time_points, values_over_time,
                    color=colors[tdoa_type], label=tdoa_type, linewidth=2)

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Correlation weights over time
        ax2.set_title('Correlation Quality (Weights) Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Average Weight')

        for weight_type in colors.keys():
            weights_over_time = []
            for weight_snapshot in self.weight_history:
                if weight_type in weight_snapshot:
                    avg_weight = np.mean(weight_snapshot[weight_type])
                    weights_over_time.append(avg_weight)
                else:
                    weights_over_time.append(np.nan)

            ax2.plot(self.time_points, weights_over_time,
                    color=colors[weight_type], label=weight_type, linewidth=2)

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Localization error over time
        ax3.set_title('Localization Error Over Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Error (meters)')

        ax3.plot(self.time_points, self.error_history, 'purple', linewidth=2)
        ax3.axhline(y=np.mean(self.error_history), color='red', linestyle='--',
                   alpha=0.7, label=f'Mean: {np.mean(self.error_history):.1f}m')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Position estimates over time
        ax4.set_title('Position Estimates Over Time')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Position (meters)')

        true_pos = self.system.drone_signal_generator.get_position()

        x_positions = [pos[0] for pos in self.position_history]
        y_positions = [pos[1] for pos in self.position_history]

        ax4.plot(self.time_points, x_positions, 'blue', linewidth=2, label='X position')
        ax4.plot(self.time_points, y_positions, 'green', linewidth=2, label='Y position')
        ax4.axhline(y=true_pos[0], color='blue', linestyle='--', alpha=0.7, label=f'True X: {true_pos[0]}')
        ax4.axhline(y=true_pos[1], color='green', linestyle='--', alpha=0.7, label=f'True Y: {true_pos[1]}')

        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Real-time TDOA Analysis using WAV File (20 seconds)', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("REAL-TIME TDOA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total analysis duration: {self.time_points[-1] - self.time_points[0]:.1f} seconds")
        print(f"Number of time windows: {len(self.time_points)}")
        print(f"Window size: {self.window_duration} seconds")
        print(f"Mean error: {np.mean(self.error_history):.1f}m ± {np.std(self.error_history):.1f}m")
        print(f"Min error: {np.min(self.error_history):.1f}m")
        print(f"Max error: {np.max(self.error_history):.1f}m")

def main():
    print("Real-time TDOA Response Test using WAV File")
    print("="*60)

    # Create and run test
    test = RealTimeTDOATest()
    test.run_realtime_analysis()
    test.plot_realtime_results()

if __name__ == "__main__":
    main()