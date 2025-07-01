#!/usr/bin/env python3
"""
Demo script for the Propellant Measurement System.
Shows how to use the system for designing a magnetic piston position sensor.
"""

import numpy as np
import matplotlib.pyplot as plt
from propellant_measurement_system import PropellantMeasurementSystem, SENSOR_DATABASE, MAGNET_DATABASE


def print_system_overview():
    """Print an overview of the system capabilities."""
    print("=" * 60)
    print("PROPELLANT MEASUREMENT SYSTEM DEMO")
    print("=" * 60)
    print("\nThis system helps design magnetic sensors for measuring")
    print("piston position in space propellant tanks.")
    print("\nKey Features:")
    print("  - Real sensor database with specifications")
    print("  - Multiple magnet types and configurations")
    print("  - Single magnet or ring of magnets")
    print("  - Digital and analog sensor outputs")
    print("  - Real-time visualization and monitoring")
    print("  - Magnetic field line visualization")
    print("  - Position accuracy analysis")
    print("  - Propellant mass estimation (MON25 and MMH)")
    print("  - Temperature and environmental effects")
    print("  - Interactive magpylib slider control")
    print("\n" + "=" * 60)


def show_sensor_comparison():
    """Display comparison of available sensors."""
    print("\n📡 AVAILABLE SENSORS:")
    print("-" * 60)
    
    for name, specs in SENSOR_DATABASE.items():
        print(f"\n{name}:")
        print(f"  Type: {specs['type']}")
        print(f"  Sensitivity: {specs['sensitivity']} {('mT' if specs['type'] == 'digital' else 'mV/mT')}")
        print(f"  Voltage Range: {specs['voltage_range'][0]}-{specs['voltage_range'][1]}V")
        print(f"  Description: {specs['description']}")


def show_magnet_comparison():
    """Display comparison of available magnets."""
    print("\n🧲 AVAILABLE MAGNETS:")
    print("-" * 60)
    
    for name, specs in MAGNET_DATABASE.items():
        print(f"\n{name}:")
        print(f"  Remanence: {specs['remanence']} T")
        print(f"  Energy Product: {specs['max_energy_product']} MGOe")
        print(f"  Temperature Coefficient: {specs['temp_coefficient']}%/°C")


def calculate_system_requirements():
    """Calculate basic system requirements."""
    print("\n📊 SYSTEM REQUIREMENTS CALCULATION:")
    print("-" * 60)
    
    # Tank specifications
    tank_length = 500  # mm
    tank_diameter = 100  # mm
    target_accuracy = tank_length * 0.1  # 10% of tank length
    
    print(f"\nTank Specifications:")
    print(f"  Length: {tank_length} mm")
    print(f"  Diameter: {tank_diameter} mm")
    print(f"  Target Accuracy: ±{target_accuracy} mm (10%)")
    
    # Calculate required sensors
    print(f"\nSensor Requirements:")
    
    for sensor_type in ["OMH3040S", "OMH3150S", "TMAG5170"]:
        sensor_data = SENSOR_DATABASE[sensor_type]
        
        # Estimate number of sensors needed
        if sensor_data["type"] == "digital":
            num_sensors = int(np.ceil((50 / target_accuracy) ** 2))
        elif sensor_data["type"] == "3-axis":
            num_sensors = int(np.ceil((20 / target_accuracy) ** 2))
        else:
            num_sensors = int(np.ceil((30 / target_accuracy) ** 2))
        
        print(f"\n  {sensor_type} ({sensor_data['type']}):")
        print(f"    Minimum sensors needed: {num_sensors}")
        print(f"    Estimated cost: ${num_sensors * 50}")  # Rough estimate


def demonstrate_position_calculation():
    """Demonstrate position calculation from sensor readings."""
    print("\n🎯 POSITION CALCULATION DEMO:")
    print("-" * 60)
    
    # Simulate sensor readings at different positions
    positions = np.linspace(0, 500, 10)  # mm
    
    # Simple voltage vs position relationship
    # In reality, this would come from magnetic field calculations
    def voltage_from_position(pos):
        # Sigmoid-like function
        return 1 + 3 / (1 + np.exp(-(pos - 250) / 100))
    
    print("\nPosition (mm) | Voltage (V) | Estimated Pos (mm) | Error (mm)")
    print("-" * 65)
    
    for pos in positions:
        voltage = voltage_from_position(pos)
        # Add some noise
        noisy_voltage = voltage + np.random.normal(0, 0.01)
        
        # Inverse calculation (simplified)
        if noisy_voltage <= 1:
            est_pos = 0
        elif noisy_voltage >= 4:
            est_pos = 500
        else:
            # Inverse sigmoid
            est_pos = 250 - 100 * np.log((3 / (noisy_voltage - 1)) - 1)
        
        error = abs(est_pos - pos)
        print(f"{pos:12.1f} | {voltage:11.3f} | {est_pos:18.1f} | {error:10.1f}")


def create_design_recommendations():
    """Create design recommendations based on analysis."""
    print("\n💡 DESIGN RECOMMENDATIONS:")
    print("-" * 60)
    
    recommendations = [
        ("Sensor Selection", 
         "For best accuracy, use TMAG5170 (3-axis) sensors. "
         "They provide full vector information and highest resolution."),
        
        ("Sensor Count", 
         "Use at least 4-8 sensors for redundancy and improved accuracy. "
         "Place them evenly around the tank circumference."),
        
        ("Magnet Type", 
         "N52 Neodymium provides strongest field, but consider SmCo "
         "for high-temperature applications (better temp stability)."),
        
        ("Magnet Size", 
         "20-30mm diameter with 10-15mm thickness provides good "
         "field strength while minimizing weight."),
        
        ("Temperature Compensation", 
         "Include temperature sensors and calibration data "
         "to compensate for magnetic field variations."),
        
        ("Calibration", 
         "Perform multi-point calibration across full tank range "
         "at different temperatures for best accuracy."),
    ]
    
    for title, recommendation in recommendations:
        print(f"\n{title}:")
        print(f"  {recommendation}")


def run_interactive_demo():
    """Run the interactive GUI application."""
    print("\n🖥️  LAUNCHING INTERACTIVE GUI...")
    print("-" * 60)
    print("Use the GUI to:")
    print("  • Select different sensors and magnets")
    print("  • Adjust system parameters")
    print("  • View real-time field calculations")
    print("  • Analyze position accuracy")
    print("  • Export results for documentation")
    print("\nClose the GUI window when done.")
    
    # Launch the application
    app = PropellantMeasurementSystem()
    app.run()


def main():
    """Main demo function."""
    # Print overview
    print_system_overview()
    
    # Show available components
    show_sensor_comparison()
    show_magnet_comparison()
    
    # Calculate requirements
    calculate_system_requirements()
    
    # Demonstrate calculations
    demonstrate_position_calculation()
    
    # Design recommendations
    create_design_recommendations()
    
    # Ask if user wants to run interactive demo
    print("\n" + "=" * 60)
    response = input("\nWould you like to launch the interactive GUI? (y/n): ")
    
    if response.lower() == 'y':
        run_interactive_demo()
    else:
        print("\nDemo complete. Run this script again to launch the GUI.")


if __name__ == "__main__":
    main() 