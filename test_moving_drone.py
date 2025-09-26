#!/usr/bin/env python3
"""
Test script for moving drone localization with animated visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drone_localization_system import DroneLocalizationSystem
import time

class MovingDroneSimulation:
    def __init__(self, config_path='config.json'):
        self.system = DroneLocalizationSystem(config_path)
        self.trajectory_data = []
        self.estimation_data = []
        self.time_data = []

    def simulate_trajectory(self, start_pos, end_pos, num_steps=20, duration=10.0):
        """Simulate drone moving from start_pos to end_pos over time."""
        print(f"Simulating drone trajectory from {start_pos} to {end_pos}")
        print(f"Steps: {num_steps}, Duration: {duration}s")

        # Generate trajectory points
        trajectory = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            current_pos = np.array(start_pos) + t * (np.array(end_pos) - np.array(start_pos))
            timestamp = t * duration
            trajectory.append((timestamp, current_pos))

        return trajectory

    def run_moving_simulation(self, trajectory):
        """Run localization for each point in trajectory."""
        self.trajectory_data.clear()
        self.estimation_data.clear()
        self.time_data.clear()

        for timestamp, true_pos in trajectory:
            print(f"Time: {timestamp:.1f}s, Position: ({true_pos[0]:.1f}, {true_pos[1]:.1f})")

            # Update drone position
            self.system.update_drone_position(true_pos.tolist())

            # Run localization
            _, estimated_pos, metrics = self.system.run_simulation()

            # Store data
            self.trajectory_data.append(true_pos)
            self.estimation_data.append(estimated_pos)
            self.time_data.append(timestamp)

            error = np.linalg.norm(estimated_pos - true_pos)
            print(f"  Estimated: ({estimated_pos[0]:.1f}, {estimated_pos[1]:.1f}), Error: {error:.1f}m")

    def create_animated_plot(self, save_gif=False):
        """Create animated visualization of moving drone localization."""
        # Setup figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Animation control variables
        self.is_paused = False

        # Get array positions for reference
        all_positions = self.system.microphone_array.get_all_positions()

        # Plot 1: Trajectory animation
        ax1.set_title('Real-time Drone Localization')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot arrays
        colors = {'array_1': 'blue', 'array_2': 'purple'}
        markers = {'array_1': '^', 'array_2': 's'}

        for array_name, positions in all_positions.items():
            color = colors.get(array_name, 'blue')
            marker = markers.get(array_name, '^')
            ax1.scatter(positions[:, 0], positions[:, 1],
                       c=color, s=100, marker=marker, label=f'{array_name}', zorder=5)

        # Initialize trajectory lines and points
        true_line, = ax1.plot([], [], 'r-', linewidth=2, alpha=0.7, label='True Path')
        est_line, = ax1.plot([], [], 'g-', linewidth=2, alpha=0.7, label='Estimated Path')
        true_point = ax1.scatter([], [], c='red', s=150, marker='o',
                               facecolors='none', edgecolors='red', linewidth=3,
                               label='Current True', zorder=10)
        est_point = ax1.scatter([], [], c='green', s=150, marker='x',
                              linewidth=3, label='Current Estimate', zorder=10)

        ax1.legend()

        # Set axis limits
        all_x = [pos[0] for traj in self.trajectory_data for pos in [traj]] + \
                [pos[0] for est in self.estimation_data for pos in [est]] + \
                [pos[0] for positions in all_positions.values() for pos in positions]
        all_y = [pos[1] for traj in self.trajectory_data for pos in [traj]] + \
                [pos[1] for est in self.estimation_data for pos in [est]] + \
                [pos[1] for positions in all_positions.values() for pos in positions]

        margin = 50
        ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # Plot 2: Error over time
        ax2.set_title('Localization Error Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Error (meters)')
        ax2.grid(True, alpha=0.3)

        # Calculate errors
        errors = [np.linalg.norm(est - true) for est, true in
                 zip(self.estimation_data, self.trajectory_data)]

        error_line, = ax2.plot([], [], 'r-', linewidth=2, label='Localization Error')
        current_error_point = ax2.scatter([], [], c='red', s=100, zorder=5)

        ax2.set_xlim(0, max(self.time_data))
        ax2.set_ylim(0, max(errors) * 1.1)
        ax2.legend()

        # Keyboard event handler
        def on_key_press(event):
            if event.key == ' ':  # Spacebar
                self.is_paused = not self.is_paused
                status = "PAUSED" if self.is_paused else "PLAYING"
                fig.suptitle(f'Moving Drone Localization - {status} (Press SPACEBAR to pause/resume)', fontsize=14)
                fig.canvas.draw()

        # Connect keyboard event
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Animation function
        def animate(frame):
            if self.is_paused:
                return true_line, est_line, true_point, est_point, error_line, current_error_point

            # Update trajectory plot
            if frame > 0:
                # True path
                true_x = [pos[0] for pos in self.trajectory_data[:frame+1]]
                true_y = [pos[1] for pos in self.trajectory_data[:frame+1]]
                true_line.set_data(true_x, true_y)

                # Estimated path
                est_x = [pos[0] for pos in self.estimation_data[:frame+1]]
                est_y = [pos[1] for pos in self.estimation_data[:frame+1]]
                est_line.set_data(est_x, est_y)

                # Current points
                true_point.set_offsets([self.trajectory_data[frame]])
                est_point.set_offsets([self.estimation_data[frame]])

                # Update error plot
                current_times = self.time_data[:frame+1]
                current_errors = errors[:frame+1]
                error_line.set_data(current_times, current_errors)
                current_error_point.set_offsets([[self.time_data[frame], errors[frame]]])

                # Update title with current info
                current_error = errors[frame]
                pause_status = "PAUSED" if self.is_paused else "PLAYING"
                ax1.set_title(f'Real-time Drone Localization (t={self.time_data[frame]:.1f}s, Error={current_error:.1f}m)')

            return true_line, est_line, true_point, est_point, error_line, current_error_point

        # Create animation
        frames = len(self.trajectory_data)
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                     interval=500, blit=False, repeat=True)

        # Set initial title
        fig.suptitle('Moving Drone Localization - Press SPACEBAR to pause/resume', fontsize=14)

        if save_gif:
            anim.save('drone_tracking_animation.gif', writer='pillow', fps=2)

        plt.tight_layout()
        plt.show()

        return anim

def test_straight_line_trajectory():
    """Test drone moving in straight line."""
    print("="*60)
    print("Moving Drone Simulation - Straight Line")
    print("="*60)

    sim = MovingDroneSimulation()

    # Define trajectory: moving from left to right
    start_pos = [50, 100]
    end_pos = [350, 200]
    trajectory = sim.simulate_trajectory(start_pos, end_pos, num_steps=15)

    # Run simulation
    sim.run_moving_simulation(trajectory)

    # Create animated visualization
    anim = sim.create_animated_plot(save_gif=True)

    # Print summary statistics
    errors = [np.linalg.norm(est - true) for est, true in
             zip(sim.estimation_data, sim.trajectory_data)]

    print(f"\nSummary Statistics:")
    print(f"Mean error: {np.mean(errors):.1f}m ± {np.std(errors):.1f}m")
    print(f"Min error: {np.min(errors):.1f}m")
    print(f"Max error: {np.max(errors):.1f}m")
    print(f"Total distance: {np.linalg.norm(np.array(end_pos) - np.array(start_pos)):.1f}m")

    return sim, anim

def test_circular_trajectory():
    """Test drone moving in circular pattern."""
    print("="*60)
    print("Moving Drone Simulation - Circular Path")
    print("="*60)

    sim = MovingDroneSimulation()

    # Define circular trajectory
    center = [150, 100]
    radius = 100
    num_steps = 20

    trajectory = []
    for i in range(num_steps):
        angle = 2 * np.pi * i / num_steps
        pos = [center[0] + radius * np.cos(angle),
               center[1] + radius * np.sin(angle)]
        timestamp = i * 0.5  # 0.5 second intervals
        trajectory.append((timestamp, np.array(pos)))

    # Run simulation
    sim.run_moving_simulation(trajectory)

    # Create animated visualization
    anim = sim.create_animated_plot(save_gif=True)

    # Print summary
    errors = [np.linalg.norm(est - true) for est, true in
             zip(sim.estimation_data, sim.trajectory_data)]

    print(f"\nCircular Path Statistics:")
    print(f"Mean error: {np.mean(errors):.1f}m ± {np.std(errors):.1f}m")
    print(f"Circle radius: {radius}m")
    print(f"Circle center: {center}")

    return sim, anim

if __name__ == "__main__":
    print("Starting Moving Drone Localization Tests...")

    # Test straight line movement
    sim1, anim1 = test_straight_line_trajectory()

    # Test circular movement
    # sim2, anim2 = test_circular_trajectory()

    print("\nAll moving drone tests completed!")