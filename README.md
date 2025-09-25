# Drone Localization using TDOA with Microphone Arrays

A Python-based simulation system for localizing drones using Time Difference of Arrival (TDOA) techniques with configurable microphone arrays. The system features GPS-synchronized microphones, various signal types, and comprehensive visualization capabilities.

## Features

### Core Functionality
- **TDOA-based localization**: Pure cross-correlation based positioning in 2D space
- **GPS synchronization**: All microphones sync to GPS clock with configurable drift and accuracy
- **Multiple signal types**: Sine waves, complex harmonics, or WAV file input
- **Configurable microphone properties**: Sensitivity, frequency response, dynamic range, directionality, audio transparency
- **Real-time visualization**: Plots with donut markers and microphone range indicators

### Signal Generation
- **Simple sine waves**: Configurable frequency and amplitude
- **Complex harmonics**: Fundamental frequency with configurable harmonics and amplitudes
- **WAV file support**: Load custom drone audio signatures with automatic resampling

### Microphone Array Features
- **GPS synchronization**: Realistic clock drift and sync accuracy simulation
- **Frequency response filtering**: Configurable low/high cutoff frequencies
- **Directional patterns**: Omnidirectional and cardioid patterns
- **Noise simulation**: Configurable background noise levels
- **Audio transparency**: Simulate microphone transparency effects

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd donut-phasing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the comprehensive test suite:
```bash
python example_test.py
```

Or use the system programmatically:
```python
from drone_localization_system import DroneLocalizationSystem

# Create system with default configuration
system = DroneLocalizationSystem('config.json')

# Run simulation
true_pos, estimated_pos, metrics = system.run_simulation()

# Plot results
system.plot_results(true_pos, estimated_pos)
```

## Configuration

The system is configured through `config.json`. Key parameters include:

### Microphone Configuration
```json
{
  "microphones": {
    "positions": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.5, -0.866]],
    "sensitivity": -40.0,
    "frequency_response": {
      "low_cutoff": 20.0,
      "high_cutoff": 20000.0
    },
    "dynamic_range": 120.0,
    "directionality": {
      "type": "omnidirectional",
      "pattern_coefficient": 0.0
    },
    "audio_transparency": 0.95,
    "gps_sync": {
      "enabled": true,
      "clock_drift": 1e-9,
      "sync_accuracy": 1e-8
    }
  }
}
```

### Drone Signal Configuration
```json
{
  "drone": {
    "position": [2.5, 1.5],
    "signal_type": "complex",
    "signal_config": {
      "complex": {
        "fundamental_frequency": 220.0,
        "harmonics": [1, 2, 3, 4],
        "harmonic_amplitudes": [1.0, 0.5, 0.3, 0.2],
        "phases": [0.0, 1.57, 3.14, 0.0]
      }
    }
  }
}
```

## File Structure

```
├── config.json                    # Main configuration file
├── microphone_array.py            # Microphone array implementation
├── drone_signal.py               # Drone signal generation
├── tdoa_localization.py          # TDOA positioning algorithm
├── drone_localization_system.py  # Main system integration
├── example_test.py               # Comprehensive test suite
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Components

### MicrophoneArray (`microphone_array.py`)
- **GPS Synchronization**: Simulates realistic GPS timing errors
- **Signal Processing**: Frequency filtering, directional patterns
- **Physical Modeling**: Distance-based delays, sensitivity effects

### DroneSignalGenerator (`drone_signal.py`)
- **Sine Wave Generation**: Simple sine wave with configurable frequency
- **Harmonic Generation**: Complex waveforms using fundamental + harmonics
- **WAV File Loading**: Custom audio signatures with resampling support

### TDOALocalizer (`tdoa_localization.py`)
- **Cross-Correlation**: Pure TDOA calculation using signal correlation
- **Position Estimation**: Hyperbola intersection method for 2D positioning
- **Quality Metrics**: TDOA measurements and confidence indicators

### DroneLocalizationSystem (`drone_localization_system.py`)
- **System Integration**: Coordinates all components
- **Visualization**: Plotting with donut markers and range circles
- **Configuration Management**: Dynamic parameter updates

## Test Suite

The `example_test.py` script provides comprehensive testing:

1. **Basic Localization**: Default configuration test
2. **Signal Type Comparison**: Sine vs complex harmonic signals
3. **Position Testing**: Multiple drone positions around array
4. **Noise Sensitivity**: Performance under various noise levels
5. **GPS Sync Effects**: Impact of synchronization accuracy

## Visualization Features

The plotting system includes:
- **Microphones**: Blue triangles with labels
- **True Position**: Red donut (hollow circle) marker
- **Estimated Position**: Green X marker
- **Error Line**: Dashed line showing position error
- **Microphone Ranges**: Dashed circles showing detection ranges
- **Grid and Labels**: Professional plot formatting

## Configuration Parameters

### Microphone Parameters
- `positions`: Array positions in meters [[x1,y1], [x2,y2], ...]
- `sensitivity`: Microphone sensitivity in dB
- `frequency_response`: Low/high cutoff frequencies in Hz
- `dynamic_range`: Dynamic range in dB
- `directionality`: Pattern type and coefficients
- `audio_transparency`: Signal transmission factor (0-1)
- `gps_sync`: GPS timing configuration

### Drone Parameters
- `position`: Drone position [x, y] in meters
- `signal_type`: "sine", "complex", or "wav_file"
- `signal_config`: Type-specific parameters

### Simulation Parameters
- `sample_rate`: Audio sample rate (Hz)
- `duration`: Simulation duration (seconds)
- `noise_level`: Background noise amplitude
- `speed_of_sound`: Sound propagation speed (m/s)

## Algorithm Details

The TDOA localization uses:
1. **Cross-correlation** to find time delays between microphone pairs
2. **Hyperbola intersection** method for position estimation
3. **Least-squares optimization** to minimize TDOA errors
4. **GPS synchronization** effects for realistic timing errors

## Limitations

- **2D positioning only**: Height (z-axis) is not considered
- **No optimization algorithms**: Pure TDOA without advanced filtering
- **Simulation only**: Not connected to real hardware
- **Static positioning**: Drone position assumed constant during measurement

## Future Enhancements

Potential improvements could include:
- 3D positioning capability
- Kalman filtering for dynamic tracking
- Multiple drone support
- Real-time hardware interface
- Advanced signal processing (beamforming, etc.)
- Machine learning-based improvements

## License

This project is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.