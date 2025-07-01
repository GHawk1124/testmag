#!/usr/bin/env python3
"""
Simple test for the propellant measurement system.
"""

from propellant_measurement_system import PropellantMeasurementSystem

def test_basic_functionality():
    """Test basic system functionality."""
    print("Testing Propellant Measurement System")
    print("=" * 50)
    
    # Create application
    app = PropellantMeasurementSystem()
    
    # Set some basic values
    app.set_value("sensor_type", "TMAG5170")
    app.set_value("num_sensors", "4")
    app.set_value("magnet_diameter", "25")
    app.set_value("magnet_length", "15")
    app.set_value("tank_id", "150")
    app.set_value("piston_position", "250")
    
    # Update system
    app.update_system()
    
    print("\nSystem created successfully!")
    print(f"- Number of sensors: {len(app.sensors)}")
    print(f"- Tank ID: {app.tank_id * 1000:.0f} mm")
    print(f"- Magnet type: {app.magnet.__class__.__name__}")
    
    # Test sensor positions
    print("\nSensor positions (stationary along tank):")
    for i, sensor in enumerate(app.sensors):
        pos = sensor.position
        print(f"  Sensor {i+1}: x={pos[0]*1000:.1f}mm, y={pos[1]*1000:.1f}mm, z={pos[2]*1000:.1f}mm")
    
    # Test field calculation at one sensor
    print("\nTesting field calculation...")
    B = app.magnet.getB(app.sensors[0].sensor_obj)
    print(f"Field at sensor 1: {B[0]*1000:.2f}, {B[1]*1000:.2f}, {B[2]*1000:.2f} mT")
    print(f"Field magnitude: {(B[0]**2 + B[1]**2 + B[2]**2)**0.5 * 1000:.2f} mT")
    
    # Test voltage calculation
    voltage = app.calculate_sensor_output(app.sensors[0], B)
    print(f"Sensor 1 voltage output: {voltage:.3f} V")
    
    print("\nBasic test completed successfully!")
    return app


if __name__ == "__main__":
    # Run basic test
    app = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("Starting GUI...")
    print("Try the following:")
    print("1. Click 'Run Analysis' to see sensor outputs")
    print("2. Click 'Show Field Grid' to visualize the magnetic field")
    print("3. Change sensor count or magnet size and click 'Update System'")
    print("4. Try 'Ring of Magnets' configuration")
    
    # Start GUI
    app.run() 