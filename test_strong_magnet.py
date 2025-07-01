#!/usr/bin/env python3
"""
Test script for strong magnet configuration to see 3D streamline tubes in PyVista.
"""

import numpy as np
from propellant_measurement_system import PropellantMeasurementSystem

def test_strong_magnet_streamlines():
    """Test PyVista with a very strong magnet to see streamline tubes."""
    print("Testing Strong Magnet Configuration for 3D Streamlines")
    print("=" * 60)
    
    # Create system with very strong magnet close to sensors
    system = PropellantMeasurementSystem()
    
    # Configure for maximum field strength
    system.set_value("magnet_config", "Single Magnet")
    system.set_value("magnet_type", "N52 Neodymium")  # Strongest magnet
    system.set_value("magnet_diameter", "50")         # Large diameter
    system.set_value("magnet_length", "30")           # Long magnet
    system.set_value("magnet_orientation", "Axial")   # Axial for strong field
    
    # Position close to sensors for maximum field
    system.set_value("piston_position", "150")        # Close to sensors
    system.set_value("tank_length", "400")            # Shorter tank
    system.set_value("tank_id", "100")                # Smaller tank
    system.set_value("num_sensors", "4")
    
    # Update system
    system.update_system()
    
    print(f"Magnet: {system.get_value('magnet_type')}")
    print(f"Size: {system.get_value('magnet_diameter')}mm x {system.get_value('magnet_length')}mm")
    print(f"Position: {system.get_value('piston_position')}mm")
    print(f"Tank: {system.get_value('tank_id')}mm ID x {system.get_value('tank_length')}mm")
    
    # Check field strength at sensors
    print("\nField strength at sensors:")
    total_field = 0
    for i, sensor in enumerate(system.sensors):
        B = system.magnet.getB(sensor.sensor_obj)
        field_mag = np.linalg.norm(B) * 1000  # mT
        total_field += field_mag
        voltage = system.calculate_sensor_output(sensor, B)
        print(f"  Sensor {i+1}: {field_mag:.2f} mT, {voltage:.3f} V")
    
    avg_field = total_field / len(system.sensors)
    print(f"\nAverage field: {avg_field:.2f} mT")
    
    if avg_field > 5:
        print("✓ Strong field detected - should produce good streamlines!")
    elif avg_field > 1:
        print("~ Moderate field - may produce streamlines")
    else:
        print("⚠ Weak field - may need adjustment")
    
    # Test PyVista visualization
    print("\n" + "=" * 60)
    print("Testing PyVista 3D Streamlines...")
    print("=" * 60)
    
    try:
        system.show_pyvista_field_lines()
        print("✓ PyVista visualization launched successfully")
        print("Look for:")
        print("  - Blue/cyan streamlines showing field lines")  
        print("  - Magnet visualization (colored object)")
        print("  - Red sensor spheres")
        print("  - Gray wireframe tank")
        
    except Exception as e:
        print(f"✗ PyVista visualization failed: {e}")
    
    return system

def test_ring_magnet_streamlines():
    """Test ring configuration for interesting streamline patterns.""" 
    print("\n" + "=" * 60)
    print("Testing Ring Magnet Configuration")
    print("=" * 60)
    
    system = PropellantMeasurementSystem()
    
    # Configure ring of strong magnets
    system.set_value("magnet_config", "Ring of Magnets")
    system.set_value("magnet_type", "N52 Neodymium")
    system.set_value("ring_num_magnets", "6")
    system.set_value("ring_diameter", "60")           # Larger ring
    system.set_value("magnet_diameter", "20")         # Good size magnets
    system.set_value("magnet_length", "15")
    system.set_value("magnet_orientation", "Radial")  # Radial for interesting patterns
    
    system.set_value("piston_position", "200")
    system.set_value("tank_length", "500")
    system.set_value("tank_id", "120")
    system.set_value("num_sensors", "6")
    
    system.update_system()
    
    print(f"Ring: {system.get_value('ring_num_magnets')} x {system.get_value('magnet_type')}")
    print(f"Ring diameter: {system.get_value('ring_diameter')}mm")
    print(f"Magnet size: {system.get_value('magnet_diameter')}mm x {system.get_value('magnet_length')}mm")
    
    # Check field
    total_field = 0
    for i, sensor in enumerate(system.sensors):
        B = system.magnet.getB(sensor.sensor_obj)
        field_mag = np.linalg.norm(B) * 1000
        total_field += field_mag
        if i == 0:
            print(f"Field at sensor 1: {field_mag:.2f} mT")
    
    avg_field = total_field / len(system.sensors)
    print(f"Average field: {avg_field:.2f} mT")
    
    print("\nTesting ring streamlines...")
    try:
        system.show_pyvista_field_lines()
        print("✓ Ring streamlines visualization launched")
        print("Look for complex field patterns from the ring configuration!")
        
    except Exception as e:
        print(f"✗ Ring visualization failed: {e}")
    
    return system

if __name__ == "__main__":
    print("Strong Magnet Streamline Test")
    print("This test creates very strong magnetic field configurations")
    print("to maximize the chances of seeing 3D streamline tubes in PyVista.\n")
    
    # Test single strong magnet
    system1 = test_strong_magnet_streamlines()
    
    # Test ring configuration  
    system2 = test_ring_magnet_streamlines()
    
    print("\n" + "=" * 60)
    print("STRONG MAGNET TEST COMPLETE")
    print("=" * 60)
    print("If you still don't see streamline tubes, the field may be too weak")
    print("for PyVista's streamline algorithm, but you should see streamlines as lines.")
    print("The magnetic fields in propellant tanks are typically quite weak,")
    print("so this is normal behavior for realistic sensor systems.") 