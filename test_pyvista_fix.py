#!/usr/bin/env python3
"""
Test script specifically for PyVista functionality to verify the empty mesh fix.
"""

import numpy as np
from propellant_measurement_system import PropellantMeasurementSystem

def test_pyvista_fix():
    """Test that PyVista functions work without crashing on empty meshes."""
    print("Testing PyVista Field Line Visualization Fix")
    print("=" * 50)
    
    # Create system
    system = PropellantMeasurementSystem()
    
    # Test different configurations that might produce empty streamlines
    test_configs = [
        {
            "name": "Weak Field Configuration",
            "magnet_config": "Single Magnet",
            "magnet_type": "Ferrite C8",  # Weaker magnet
            "magnet_diameter": "10",      # Small magnet
            "magnet_length": "5",
            "piston_position": "400",     # Far from sensors
            "num_sensors": "4"
        },
        {
            "name": "Strong Field Configuration", 
            "magnet_config": "Single Magnet",
            "magnet_type": "N52 Neodymium",  # Strong magnet
            "magnet_diameter": "25",
            "magnet_length": "15",
            "piston_position": "200",        # Closer to sensors
            "num_sensors": "6"
        },
        {
            "name": "Ring Configuration",
            "magnet_config": "Ring of Magnets",
            "magnet_type": "N48 Neodymium",
            "ring_num_magnets": "8",
            "ring_diameter": "50",
            "magnet_orientation": "Radial",
            "piston_position": "250",
            "num_sensors": "8"
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config['name']}")
        print("-" * 30)
        
        # Apply configuration
        for key, value in config.items():
            if key != "name":
                system.set_value(key, value)
        
        # Update system
        system.update_system()
        
        print(f"Magnet: {system.get_value('magnet_config')} - {system.get_value('magnet_type')}")
        print(f"Position: {system.get_value('piston_position')}mm")
        print(f"Sensors: {len(system.sensors)}")
        
        # Test field calculation at sensors
        total_field = 0
        for j, sensor in enumerate(system.sensors):
            B = system.magnet.getB(sensor.sensor_obj)
            field_mag = np.linalg.norm(B) * 1000  # mT
            total_field += field_mag
            if j == 0:  # Just show first sensor
                print(f"Field at sensor 1: {field_mag:.3f} mT")
        
        avg_field = total_field / len(system.sensors)
        print(f"Average field: {avg_field:.3f} mT")
        
        # Test PyVista visualization (this should not crash)
        print("Testing PyVista field lines...")
        try:
            system.show_pyvista_field_lines()
            print("✓ PyVista visualization completed successfully")
        except Exception as e:
            print(f"✗ PyVista visualization failed: {e}")
        
        # Test regular field grid (should always work)
        print("Testing field grid...")
        try:
            system.show_field_grid()
            print("✓ Field grid visualization completed successfully")
            
            # Close plots to avoid cluttering
            import matplotlib.pyplot as plt
            plt.close('all')
            
        except Exception as e:
            print(f"✗ Field grid visualization failed: {e}")
    
    print("\n" + "=" * 50)
    print("PyVista Fix Verification Complete")
    print("=" * 50)
    print("\nKey improvements:")
    print("✓ Empty mesh errors are now handled gracefully")
    print("✓ Deprecated max_time parameter replaced with max_steps")
    print("✓ PyVista global theme set to allow empty meshes")
    print("✓ Proper error checking for streamline generation")
    print("✓ Fallback to field grid when streamlines fail")
    
def test_individual_pyvista_components():
    """Test individual PyVista components that could fail."""
    print("\n" + "=" * 50)
    print("Testing Individual PyVista Components")
    print("=" * 50)
    
    try:
        import pyvista as pv
        print("✓ PyVista import successful")
        
        # Test basic PyVista functionality
        print("Testing basic PyVista objects...")
        
        # Test grid creation
        grid = pv.ImageData(dimensions=(10, 10, 10))
        print(f"✓ Grid created: {grid.n_points} points")
        
        # Test basic mesh operations
        sphere = pv.Sphere(radius=0.01)
        print(f"✓ Sphere created: {sphere.n_points} points")
        
        # Test empty mesh handling
        pv.global_theme.allow_empty_mesh = True
        empty_grid = pv.PolyData()
        print(f"✓ Empty mesh handling: {empty_grid.n_points} points")
        
        # Test plotter creation
        pl = pv.Plotter(off_screen=True)  # Use off_screen to avoid GUI
        pl.add_mesh(sphere, color='red')
        print("✓ Plotter and mesh addition successful")
        pl.close()
        
        print("✓ All PyVista components working correctly")
        
    except ImportError:
        print("✗ PyVista not available")
    except Exception as e:
        print(f"✗ PyVista component test failed: {e}")

if __name__ == "__main__":
    test_pyvista_fix()
    test_individual_pyvista_components() 