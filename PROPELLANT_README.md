# Propellant Measurement System

A comprehensive GUI application for designing magnetic-based piston position sensors for space propellant tanks.

## Overview

This system helps engineers design and analyze magnetic sensor arrays for measuring piston position in propellant tanks, targeting <10% position accuracy for reliable propellant mass estimation.

## Features

### System Design
- **Sensor Database**: Real hall-effect sensors with accurate specifications
  - Digital threshold sensors (e.g., OMH3040S)
  - Single-axis magnitude sensors (e.g., OMH3150S)
  - 3-axis vector sensors (e.g., TMAG5170)
  - Digital vs Analog output selection
  - Sensors positioned at y=0 plane on cylinder surface
- **Magnet Configuration**: Multiple magnet types and materials
  - Single magnet or ring of magnets configuration
  - Neodymium (N35, N48, N52)
  - Samarium Cobalt (temperature stable)
  - Ferrite (cost-effective)
  - Axial or diametral (radial) magnetization
- **Tank Geometry**: Configurable cylinder dimensions and materials
  - Input-based controls (no sliders) for precise values

### Analysis Tools
- **Vectorized Field Computation**: Fast path-based calculations using Magpylib
- **Field Visualization**: 
  - 2D field contour plots in multiple planes
  - Field line visualization with matplotlib
  - 3D field lines using PyVista integration
  - Field grid analysis with vector plots
- **Real-time Visualization**: Live sensor readings and 3D system view
- **Position Accuracy**: Error analysis and optimization
- **Propellant Mass Estimation**: MON25 and MMH mass from sensor-estimated position
- **Temperature Effects**: Environmental impact on accuracy
- **Signal-to-Noise Ratio**: SNR analysis for different configurations
- **Cost vs Accuracy**: Trade-off analysis for system optimization

### Interactive Features
- Input-controlled parameters (exact values, no sliders in GUI)
- Magpylib slider for interactive piston position control
- Real-time magnetic field calculation
- Individual sensor output monitoring
- Animation of piston movement with proper paths
- Export capabilities for documentation

## Installation

1. Ensure you have the MagUI library installed (see main README)
2. The propellant system files are included in the `magui` directory

## Quick Start

### Run the Demo
```bash
cd magui
python propellant_demo.py
```

This will:
1. Show available sensors and magnets
2. Calculate system requirements
3. Demonstrate position calculations
4. Provide design recommendations
5. Optionally launch the interactive GUI

### Direct GUI Launch
```bash
cd magui
python propellant_measurement_system.py
```

## Usage Guide

### 1. Sensor Configuration
- Select a sensor model from the dropdown (or choose "Custom")
- The system automatically updates sensitivity and voltage range
- Adjust the number of sensors (1-20)
- Sensors are automatically placed at y=0, z=tangent to cylinder

### 2. Magnet Setup
- Choose magnet material (affects field strength and temperature stability)
- Select shape: Cylinder or Disc
- Adjust dimensions with sliders
- Set magnetization direction (Axial or Diametral)

### 3. Tank Configuration
- Set inner diameter (50-500mm)
- Define wall thickness (1-20mm)
- Select material (affects sensor mounting)

### 4. Analysis Options

#### Run Analysis
Generates comprehensive plots:
- All sensor outputs vs position
- Average sensor response
- Position estimation error
- Relative error percentage

#### Animate System
Shows 3D animation of:
- Piston movement
- Sensor readings in real-time
- Magnetic field visualization

#### Real-time Monitor
Opens separate window with:
- Live 3D system view
- Current sensor outputs
- Position tracking over time
- Error histogram
- Adjustable update rate and noise

#### Advanced Analysis
Provides optimization plots:
- Sensor sensitivity comparison
- Optimal sensor count
- Magnet size effects
- Temperature impacts
- SNR analysis
- Cost-accuracy trade-offs

## System Design Guidelines

### Target: <10% Position Accuracy

1. **Sensor Selection**
   - For highest accuracy: Use TMAG5170 (3-axis) sensors
   - For cost-effectiveness: Use A1324 (high sensitivity linear)
   - For simple threshold detection: Use OMH3040S (digital)

2. **Sensor Placement**
   - Minimum 4 sensors for redundancy
   - 8-12 sensors optimal for 10% accuracy
   - Even spacing around circumference
   - All at y=0, z=tangent as specified

3. **Magnet Selection**
   - N52 for maximum field strength
   - SmCo for temperature stability (-40°C to +85°C)
   - 20-30mm diameter recommended
   - 10-15mm thickness for adequate field

4. **Calibration Requirements**
   - Multi-point calibration across full range
   - Temperature compensation tables
   - Individual sensor calibration
   - System-level verification

## Output Interpretation

### Voltage vs Position
- Should show monotonic relationship
- Steeper slope = better sensitivity
- Watch for saturation at extremes

### Position Error Analysis
- Absolute error in mm
- Relative error as percentage
- Must stay below 50mm for 10% target (500mm tank)

### Real-time Monitoring
- Green bars: Good signal strength
- Check for sensor consistency
- Monitor noise levels
- Verify all sensors responding

## Export Options

### JSON Export
Includes:
- Complete system configuration
- Sensor and magnet databases
- Analysis timestamp
- All parameter values

### Future: CSV Export
Will include:
- Position vs voltage tables
- Calibration data
- Error statistics

## Troubleshooting

### Poor Accuracy
- Increase number of sensors
- Use higher sensitivity sensors
- Increase magnet size
- Check for magnetic interference

### Inconsistent Readings
- Verify sensor alignment
- Check for temperature variations
- Ensure proper grounding
- Validate sensor connections

### No Signal
- Check magnetization direction
- Verify sensor power
- Confirm magnet material properties
- Test individual sensors

## Advanced Features

### Custom CAD Import
- Load STL/STEP files for magnet geometry
- Useful for complex magnet shapes
- Requires additional CAD processing

### Hysteresis Modeling
- Database includes coercivity values
- Important for repeated measurements
- Consider for high-accuracy applications

### Mass Estimation
- Automatic calculation from position
- Assumes uniform propellant density
- Configurable for different propellants

## Example Configurations

### High Accuracy Setup
- Sensors: 8x TMAG5170 (3-axis)
- Magnet: N52, 25mm dia, 12mm thick
- Expected accuracy: <5% position error

### Cost-Effective Setup
- Sensors: 6x A1324 (linear)
- Magnet: N48, 20mm dia, 10mm thick
- Expected accuracy: ~8% position error

### Minimal Setup
- Sensors: 4x OMH3150S (linear)
- Magnet: N35, 20mm dia, 10mm thick
- Expected accuracy: ~10% position error

## Future Enhancements

1. **Flex PCB Design Export**
   - Generate sensor layout for PCB
   - Include mounting features
   - Export to CAD formats

2. **Advanced Calibration**
   - Automated calibration routines
   - Machine learning position estimation
   - Adaptive filtering

3. **Environmental Testing**
   - Vibration effects
   - Radiation tolerance
   - Thermal cycling simulation

4. **Data Integration**
   - Direct sensor data import
   - Real hardware interface
   - Telemetry integration

## References

- Hall Effect Sensor Application Notes
- Magnetic Field Calculation Theory
- Space Propulsion System Design
- Temperature Compensation Techniques

---

For questions or contributions, see the main MagUI documentation. 