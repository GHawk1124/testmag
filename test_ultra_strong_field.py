#!/usr/bin/env python3
"""
Test script for ultra-strong magnetic field visualization
Demonstrates dramatic field strength for clear visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from propellant_measurement_system import PropellantMeasurementSystem
import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def test_ultra_strong_field():
    """Test ultra-strong magnetic field configuration"""
    print("🚀 Testing Ultra-Strong Magnetic Field System")
    print("=" * 50)
    
    # Create the system
    app = PropellantMeasurementSystem()
    
    # Set ultra-strong configuration
    app.set_value("magnet_type", "N52 Ultra Strong")
    app.set_value("magnet_diameter", "80")  # 80mm diameter
    app.set_value("magnet_length", "40")    # 40mm length
    app.set_value("magnet_remanence", "2.5")  # 2.5T remanence
    
    # Update system with new values
    app.update_system()
    
    # Test field strength at various positions
    print("\n📊 Field Strength Analysis:")
    print("=" * 30)
    
    # Test positions around the magnet
    test_positions = np.array([
        [0.0, 0.0, 0.01],   # Very close - 1cm
        [0.0, 0.0, 0.02],   # Close - 2cm  
        [0.0, 0.0, 0.05],   # Medium - 5cm
        [0.0, 0.0, 0.10],   # Far - 10cm
        [0.05, 0.0, 0.05],  # Off-axis
    ])
    
    if app.magnet:
        B_field = app.magnet.getB(test_positions)
        B_magnitude = np.linalg.norm(B_field, axis=1)
        
        for i, (pos, B_mag) in enumerate(zip(test_positions, B_magnitude)):
            pos_str = f"[{pos[0]*1000:.0f}, {pos[1]*1000:.0f}, {pos[2]*1000:.0f}]"
            print(f"Position {pos_str}mm: {B_mag*1000:.1f} mT")
    
    # Test sensor outputs
    print("\n🔌 Sensor Output Analysis:")
    print("=" * 30)
    
    if app.sensors:
        for i, sensor in enumerate(app.sensors[:3]):  # Show first 3 sensors
            if sensor.sensor_obj:
                field = app.magnet.getB(sensor.position)
                field_mag = np.linalg.norm(field) * 1000  # Convert to mT
                voltage = app.calculate_sensor_output(sensor, field)
                print(f"Sensor {i+1}: {field_mag:.1f} mT → {voltage:.3f} V")
    
    print(f"\n✅ System configured with ultra-strong fields!")
    print(f"   Magnet: {app.current_magnet_properties['remanence']} T remanence")
    print(f"   Max field expected: {app.current_magnet_properties['remanence']*1000:.0f} mT at contact")
    
    # Launch GUI for interactive visualization
    print(f"\n🎮 Launching GUI for interactive visualization...")
    print(f"   Try: 'Show Field Lines', '3D Field Lines (PyVista)', 'Run Analysis'")
    
    return app

def demo_pyvista_streamlines():
    """Demonstrate PyVista streamlines with ultra-strong magnets"""
    print("\n🌊 PyVista Streamlines Demo")
    print("=" * 30)
    
    try:
        # Create ultra-strong cylindrical magnet
        magnet = magpy.magnet.Cylinder(
            polarization=(0, 0, 2500),  # 2.5T in mT
            dimension=(0.080, 0.040),   # 80mm x 40mm
            position=(0, 0, 0)
        )
        
        # Create a 3D grid (adjusted for ultra-strong fields)
        grid = pv.ImageData(
            dimensions=(31, 31, 31),
            spacing=(0.005, 0.005, 0.005),  # 5mm spacing for better resolution
            origin=(-0.075, -0.075, -0.075),   # Smaller 7.5cm range due to strong fields
        )
        
        # Compute B-field
        print("Computing magnetic field on 3D grid...")
        B_field = magnet.getB(grid.points) * 1000  # Convert to mT
        grid["B"] = B_field
        
        max_field = np.max(np.linalg.norm(B_field, axis=1))
        print(f"Maximum field magnitude: {max_field:.1f} mT")
        
        # Create seed points for streamlines (adjusted for ultra-strong fields)
        seed = pv.Disc(inner=0.005, outer=0.02, r_res=2, c_res=12)
        seed.translate((0, 0, 0.025))  # Further from ultra-strong magnet
        
        print("Computing streamlines...")
        strl = grid.streamlines_from_source(
            seed,
            vectors="B",
            max_step_length=0.001,  # Smaller steps for better integration
            max_steps=2000,         # More steps allowed
            integration_direction="both",
        )
        
        print(f"Generated {len(strl.points)} streamline points")
        
        # Create visualization
        pl = pv.Plotter()
        
        # Add magnet
        magpy.show(magnet, canvas=pl, units_length="m", backend="pyvista")
        
        # Add streamlines as tubes (with fallback for ultra-strong fields)
        if len(strl.points) > 0:
            print(f"Streamlines created with {len(strl.points)} points")
            try:
                tubes = strl.tube(radius=0.0005)  # Smaller radius for ultra-strong fields
                if tubes.n_points > 0:
                    pl.add_mesh(
                        tubes,
                        cmap="bwr",
                        scalar_bar_args={
                            "title": "B Field (mT)",
                            "title_font_size": 16,
                            "color": "black",
                            "vertical": True,
                        },
                    )
                    print("✅ Streamline tubes added successfully")
                else:
                    # Fallback: draw lines instead
                    pl.add_mesh(strl, cmap="bwr", line_width=3)
                    print("✅ Streamlines added as lines (fallback)")
            except Exception as e:
                # Fallback: draw lines instead
                pl.add_mesh(strl, cmap="bwr", line_width=3)
                print(f"✅ Streamlines added as lines (tube failed: {e})")
        else:
            print("⚠️ No streamline points generated")
        
        # Set camera
        pl.camera.position = (0.15, 0.15, 0.15)
        pl.show()
        
    except Exception as e:
        print(f"❌ PyVista demo failed: {e}")
        print("Make sure PyVista is installed: pip install pyvista")

if __name__ == "__main__":
    # Run the test
    app = test_ultra_strong_field()
    
    # Ask user what to demo
    print("\n" + "="*50)
    print("Choose demo:")
    print("1. Run GUI (default)")
    print("2. PyVista Streamlines Demo")
    print("3. Both")
    choice = input("Enter choice (1-3) or press Enter for GUI: ").strip()
    
    if choice == "2":
        demo_pyvista_streamlines()
    elif choice == "3":
        demo_pyvista_streamlines()
        app.mainloop()
    else:
        # Default: just run the GUI
        app.mainloop() 