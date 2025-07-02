#!/usr/bin/env python3
"""
Propellant Measurement System GUI
A system for measuring piston position in space propellant tanks using magnetic field sensors.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import magpylib as magpy
from scipy.interpolate import interp1d
import customtkinter as ctk
from tkinter import filedialog
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from magui import MagUI


# Sensor database with real specifications
SENSOR_DATABASE = {
    "OMH3040S": {
        "type": "digital",
        "sensitivity": 40.0,  # mT threshold
        "axis": "z",
        "voltage_range": (0, 5),
        "size": (3.0, 3.0, 1.5),  # mm
        "hysteresis": 10.0,  # mT
        "description": "Digital Hall switch, omnipolar"
    },
    "OMH3150S": {
        "type": "magnitude",
        "sensitivity": 1.4,  # mV/mT
        "axis": "z",
        "voltage_range": (0.5, 4.5),
        "size": (3.0, 3.0, 1.5),  # mm
        "offset": 2.5,  # V at 0 field
        "description": "Linear Hall sensor, single axis"
    },
    "TMAG5170": {
        "type": "3-axis",
        "sensitivity": 50.0,  # LSB/mT (12-bit)
        "axis": "xyz",
        "voltage_range": (0, 3.3),
        "size": (3.0, 3.0, 0.8),  # mm
        "resolution": 12,  # bits
        "description": "3D Hall sensor, high precision"
    },
    "A1324": {
        "type": "magnitude",
        "sensitivity": 5.0,  # mV/mT
        "axis": "z",
        "voltage_range": (0, 5),
        "size": (3.0, 3.0, 1.5),  # mm
        "offset": 2.5,
        "description": "Linear Hall sensor, high sensitivity"
    },
    "MLX90393": {
        "type": "3-axis",
        "sensitivity": 0.161,  # μT/LSB
        "axis": "xyz",
        "voltage_range": (0, 3.3),
        "size": (3.0, 3.0, 0.85),  # mm
        "resolution": 16,  # bits
        "description": "Triaxis magnetic sensor"
    }
}

# Magnet database with typical properties
MAGNET_DATABASE = {
    "N52 Ultra Strong": {
        "remanence": 2.5,  # T - Ultra strong for visualization
        "coercivity": 1200000,  # A/m
        "max_energy_product": 80,  # MGOe
        "temp_coefficient": -0.12,  # %/°C
        "density": 7500,  # kg/m³
    },
    "N52 Neodymium": {
        "remanence": 1.45,  # T
        "coercivity": 907000,  # A/m
        "max_energy_product": 52,  # MGOe
        "temp_coefficient": -0.12,  # %/°C
        "density": 7500,  # kg/m³
    },
    "N48 Neodymium": {
        "remanence": 1.38,
        "coercivity": 907000,
        "max_energy_product": 48,
        "temp_coefficient": -0.12,
        "density": 7500,
    },
    "N35 Neodymium": {
        "remanence": 1.18,
        "coercivity": 868000,
        "max_energy_product": 35,
        "temp_coefficient": -0.12,
        "density": 7400,
    },
    "Ferrite C8": {
        "remanence": 0.4,
        "coercivity": 235000,
        "max_energy_product": 3.5,
        "temp_coefficient": -0.20,
        "density": 4900,
    },
    "SmCo 2:17": {
        "remanence": 1.1,
        "coercivity": 750000,
        "max_energy_product": 32,
        "temp_coefficient": -0.03,
        "density": 8400,
    }
}


@dataclass
class SensorConfig:
    """Configuration for a single sensor."""
    position: Tuple[float, float, float]
    sensor_type: str
    properties: Dict[str, Any]
    sensor_obj: Optional[magpy.Sensor] = None


class PropellantMeasurementSystem(MagUI):
    """Main application for propellant measurement system design."""
    
    def __init__(self):
        super().__init__("Propellant Measurement System Designer", (1200, 800))
        
        # System state
        self.sensors: List[SensorConfig] = []
        self.magnet = None
        self.cylinder = None
        self.current_sensor_properties = SENSOR_DATABASE["TMAG5170"].copy()
        self.current_magnet_properties = MAGNET_DATABASE["N52 Ultra Strong"].copy()
        
        # Tank properties
        self.tank_id = 0.457  # m (18" standard spacecraft tank)
        self.tank_thickness = 0.004  # m
        self.tank_length = 1.0  # m
        
        # Build GUI
        self.setup_gui()
        

        
    def setup_gui(self):
        """Setup the complete GUI."""
        # Title
        self.add_label("Propellant Tank Measurement System", font=("Arial", 20, "bold"))
        
        # Sensor Configuration Section
        self.add_label("\nSensor Configuration", font=("Arial", 16, "bold"))
        
        # Sensor selection dropdown
        sensor_names = list(SENSOR_DATABASE.keys()) + ["Custom"]
        self.add_combobox("sensor_type", sensor_names, "Sensor Model:", "TMAG5170")
        self.widgets["sensor_type"].configure(command=self.on_sensor_change)
        
        # Sensor properties (will be populated based on selection)
        self.add_entry("sensor_sensitivity", "Sensitivity:", "50.0", "mV/mT or LSB/mT")
        self.add_combobox("sensor_axis", ["z", "xyz"], "Sensing Axis:", "xyz")
        self.add_entry("sensor_voltage_min", "Min Voltage (V):", "0.0")
        self.add_entry("sensor_voltage_max", "Max Voltage (V):", "3.3")
        self.add_combobox("sensor_output_type", ["Analog", "Digital"], "Output Type:", "Analog")
        
        # Number of sensors
        self.add_entry("num_sensors", "Number of Sensors:", "8", "1-20")
        
        # Magnet Configuration Section
        self.add_label("\nMagnet Configuration", font=("Arial", 16, "bold"))
        
        # Magnet configuration type
        self.add_combobox("magnet_config", ["Single Magnet", "Ring of Magnets"], "Configuration:", "Single Magnet")
        
        # Magnet selection
        magnet_names = list(MAGNET_DATABASE.keys()) + ["Custom", "Load CAD"]
        self.add_combobox("magnet_type", magnet_names, "Magnet Type:", "N52 Ultra Strong")
        self.widgets["magnet_type"].configure(command=self.on_magnet_change)
        
        # Magnet properties
        self.add_entry("magnet_remanence", "Remanence (T):", "2.5")
        self.add_combobox("magnet_shape", ["Cylinder", "Disc", "CAD"], "Shape:", "Cylinder")
        self.add_entry("magnet_diameter", "Diameter (mm):", "80", "5-200")
        self.add_entry("magnet_length", "Length (mm):", "40", "5-200")
        self.add_combobox("magnet_orientation", ["Axial", "Diametral"], "Magnetization:", "Axial")
        
        # Ring configuration
        self.add_entry("ring_diameter", "Ring Diameter (mm):", "80", "For ring configuration")
        self.add_entry("ring_num_magnets", "Number in Ring:", "4", "2-20")
        
        # Tank Configuration Section
        self.add_label("\nTank Configuration", font=("Arial", 16, "bold"))
        
        self.add_entry("tank_id", "Inner Diameter (mm):", "457", "50-500")
        self.add_entry("tank_thickness", "Wall Thickness (mm):", "4", "1-20")
        self.add_entry("tank_length", "Tank Length (mm):", "1000", "100-2000")
        self.add_combobox("tank_material", ["Aluminum", "Titanium", "CFRP", "Steel"], 
                         "Material:", "Aluminum")
        
        # Piston Control
        self.add_label("\nPiston Control", font=("Arial", 16, "bold"))
        self.add_entry("piston_position", "Piston Position (mm):", "500", "0-1000")
        self.add_checkbox("animate_piston", "Animate Piston Movement", False)
        self.add_checkbox("use_magpy_slider", "Use Magpylib Slider Control", False)
        
        # Analysis Options
        self.add_label("\nAnalysis Options", font=("Arial", 16, "bold"))
        self.add_checkbox("show_field_lines", "Show Field Lines", True)
        self.add_checkbox("show_individual_plots", "Individual Sensor Plots", False)
        self.add_checkbox("show_accuracy_plot", "Show Position Accuracy", True)
        self.add_checkbox("show_propellant_mass", "Show Propellant Mass", True)
        
        # Action Buttons
        self.add_button("Update System", self.update_system)
        self.add_button("Run Analysis", self.run_analysis)
        self.add_button("Animate System", self.animate_system)
        self.add_button("Show Field Lines", self.show_field_visualization)
        self.add_button("Show Field Grid", self.show_field_grid)
        self.add_button("3D Field Lines (PyVista)", self.show_pyvista_field_lines)
        self.add_button("Export Results", self.export_results)
        
        # Initialize system
        self.update_system()
    
    def on_sensor_change(self, choice):
        """Handle sensor selection change."""
        if choice in SENSOR_DATABASE:
            sensor_data = SENSOR_DATABASE[choice]
            self.set_value("sensor_sensitivity", str(sensor_data["sensitivity"]))
            self.set_value("sensor_axis", sensor_data["axis"])
            self.set_value("sensor_voltage_min", str(sensor_data["voltage_range"][0]))
            self.set_value("sensor_voltage_max", str(sensor_data["voltage_range"][1]))
            self.current_sensor_properties = sensor_data.copy()
    
    def on_magnet_change(self, choice):
        """Handle magnet selection change."""
        if choice in MAGNET_DATABASE:
            magnet_data = MAGNET_DATABASE[choice]
            self.set_value("magnet_remanence", str(magnet_data["remanence"]))
            self.current_magnet_properties = magnet_data.copy()
        elif choice == "Load CAD":
            self.load_cad_file()
    
    def load_cad_file(self):
        """Load CAD file for magnet geometry."""
        filename = filedialog.askopenfilename(
            title="Select CAD file",
            filetypes=[("STL files", "*.stl"), ("STEP files", "*.step"), ("All files", "*.*")]
        )
        if filename:
            print(f"Loading CAD file: {filename}")
            # Here you would implement CAD loading logic
    
    def update_system(self):
        """Update the magnetic system based on current parameters."""
        # Get current values
        values = self.get_values()
        
        # Update tank dimensions
        self.tank_id = float(values["tank_id"]) / 1000  # Convert to meters
        self.tank_thickness = float(values["tank_thickness"]) / 1000
        self.tank_length = float(values.get("tank_length", 500)) / 1000
        
        # Create magnet(s)
        magnet_diameter = float(values["magnet_diameter"]) / 1000
        magnet_length = float(values["magnet_length"]) / 1000
        magnet_position = float(values["piston_position"]) / 1000
        
        # Magnetization direction - convert remanence (T) to polarization (T)
        remanence = float(values["magnet_remanence"])
        if values["magnet_orientation"] == "Axial":
            # Axial magnetization along cylinder axis (x-direction)
            polarization = (remanence, 0, 0)
        else:
            # Diametral magnetization (radial)
            polarization = (0, 0, remanence)
        
        # Create single magnet or ring of magnets
        if values["magnet_config"] == "Single Magnet":
            self.magnet = magpy.magnet.Cylinder(
                polarization=polarization,
                dimension=(magnet_diameter, magnet_length),
                position=(magnet_position, 0, 0)
            )
        else:
            # Ring of magnets
            ring_diameter = float(values["ring_diameter"]) / 1000
            num_magnets = int(values["ring_num_magnets"])
            
            # Create collection for ring
            self.magnet = magpy.Collection()
            
            for i in range(num_magnets):
                angle = 2 * np.pi * i / num_magnets
                # Position on ring
                y_pos = ring_diameter/2 * np.cos(angle)
                z_pos = ring_diameter/2 * np.sin(angle)
                
                # For ring configuration, adjust polarization direction
                if values["magnet_orientation"] == "Axial":
                    # All magnets point in same direction (axial)
                    mag_polarization = polarization
                else:
                    # Diametral - each magnet points radially
                    mag_polarization = (0, 
                                      remanence * np.cos(angle),
                                      remanence * np.sin(angle))
                
                mag = magpy.magnet.Cylinder(
                    polarization=mag_polarization,
                    dimension=(magnet_diameter, magnet_length),
                    position=(magnet_position, y_pos, z_pos)
                )
                
                # Orient cylinder axis along ring tangent
                mag.rotate_from_angax(angle * 180/np.pi, 'x')
                
                self.magnet.add(mag)
        
        # Create sensors
        num_sensors = int(values["num_sensors"])
        self.sensors = []
        
        # Place sensors along the tank at y=0, z=top of cylinder (tangent)
        sensor_radius = self.tank_id / 2 + self.tank_thickness
        
        # Distribute sensors evenly along x-axis (tank length)
        if num_sensors == 1:
            # Single sensor at middle of tank
            sensor_positions = [(self.tank_length / 2, 0, sensor_radius)]
        else:
            # Multiple sensors evenly distributed along tank length
            x_positions = np.linspace(0.1 * self.tank_length, 0.9 * self.tank_length, num_sensors)
            sensor_positions = []
            for x_pos in x_positions:
                # All sensors at y=0, z=top of tank
                sensor_positions.append((x_pos, 0, sensor_radius))
        
        for i, pos in enumerate(sensor_positions):
            sensor = magpy.Sensor(position=pos)
            
            # Update sensor properties including output type
            props = self.current_sensor_properties.copy()
            props["output_type"] = values.get("sensor_output_type", "Analog")
            
            self.sensors.append(SensorConfig(
                position=pos,
                sensor_type=values["sensor_type"],
                properties=props,
                sensor_obj=sensor
            ))
        
        print(f"System updated: {num_sensors} sensors, magnet at {magnet_position*1000:.1f}mm")
    
    def calculate_sensor_output(self, sensor: SensorConfig, field: np.ndarray) -> float:
        """Calculate sensor voltage output based on field and sensor type."""
        props = sensor.properties
        output_type = props.get("output_type", "Analog")
        
        # Calculate field component based on sensor axis
        if props["axis"] == "z":
            field_component = field[2] * 1000  # mT
        elif props["axis"] == "x":
            field_component = field[0] * 1000  # mT
        elif props["axis"] == "y":
            field_component = field[1] * 1000  # mT
        else:  # xyz - use magnitude
            field_component = np.linalg.norm(field) * 1000  # mT
        
        if props["type"] == "digital" or output_type == "Digital":
            # Digital sensor - threshold detection
            threshold = props.get("sensitivity", 40.0)  # mT threshold
            hysteresis = props.get("hysteresis", 10.0)  # mT
            
            # Simple threshold logic (without hysteresis tracking for now)
            if abs(field_component) > threshold:
                return props["voltage_range"][1]  # High
            else:
                return props["voltage_range"][0]  # Low
                
        else:  # Analog output
            if props["type"] == "magnitude":
                # Linear Hall sensor
                sensitivity = props["sensitivity"]  # mV/mT
                offset = props.get("offset", (props["voltage_range"][0] + props["voltage_range"][1])/2)
                
                voltage = offset + field_component * sensitivity / 1000  # Convert mV to V
                
            elif props["type"] == "3-axis":
                # 3-axis sensor with analog output
                sensitivity = props.get("sensitivity", 50.0)  # LSB/mT or mV/mT
                
                # For analog output mode, convert to voltage
                v_min, v_max = props["voltage_range"]
                max_field = 50.0  # mT typical range
                
                # Linear scaling
                voltage = v_min + (v_max - v_min) * (field_component / max_field)
                
            else:
                # Default analog behavior
                sensitivity = 1.0  # mV/mT default
                offset = 2.5
                voltage = offset + field_component * sensitivity / 1000
            
            # Clamp to voltage range
            voltage = np.clip(voltage, props["voltage_range"][0], props["voltage_range"][1])
            return voltage
    
    def run_analysis(self):
        """Run the magnetic field analysis and display results using vectorized computation."""
        if not self.magnet or not self.sensors:
            print("Please update system first!")
            return
        
        # Create figure with subplots
        if self.get_value("show_individual_plots"):
            num_sensors = len(self.sensors)
            cols = int(np.ceil(np.sqrt(num_sensors)))
            rows = int(np.ceil(num_sensors / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            axes = axes.flatten() if num_sensors > 1 else [axes]
        else:
            # Create 2x3 grid for standard analysis
            if self.get_value("show_propellant_mass"):
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Create position path for piston movement
        positions = np.linspace(0, self.tank_length, 100)
        
        # Set up magnet path
        if isinstance(self.magnet, magpy.Collection):
            # For collection, set path for each magnet
            for mag in self.magnet:
                y_pos = mag.position[1]
                z_pos = mag.position[2]
                path = np.column_stack([positions, np.full_like(positions, y_pos), np.full_like(positions, z_pos)])
                mag.position = path
        else:
            # Single magnet path
            path = np.column_stack([positions, np.zeros_like(positions), np.zeros_like(positions)])
            self.magnet.position = path
        
        # Sensors stay stationary - no paths!
        sensor_objs = []
        for sensor in self.sensors:
            # Keep sensor at its fixed position
            sensor.sensor_obj.position = sensor.position
            sensor_objs.append(sensor.sensor_obj)
        
        # Vectorized field computation - magnet moves, sensors stay fixed
        B_fields = self.magnet.getB(sensor_objs)  # Shape: (num_sensors, num_positions, 3)
        
        # Calculate sensor outputs
        sensor_outputs = np.zeros((len(positions), len(self.sensors)))
        
        for i, sensor in enumerate(self.sensors):
            for j, B in enumerate(B_fields[i]):
                voltage = self.calculate_sensor_output(sensor, B)
                sensor_outputs[j, i] = voltage
        
        # Plot results
        if self.get_value("show_individual_plots"):
            # Individual sensor plots
            for i, ax in enumerate(axes[:len(self.sensors)]):
                ax.plot(positions * 1000, sensor_outputs[:, i])
                ax.set_xlabel("Piston Position (mm)")
                ax.set_ylabel("Voltage (V)")
                ax.set_title(f"Sensor {i+1}")
                ax.grid(True, alpha=0.3)
        else:
            # Combined plots
            axes_flat = axes.flatten()
            ax_idx = 0
            
            # All sensors on one plot
            ax1 = axes_flat[ax_idx]
            ax_idx += 1
            for i in range(len(self.sensors)):
                ax1.plot(positions * 1000, sensor_outputs[:, i], label=f"Sensor {i+1}")
            ax1.set_xlabel("Piston Position (mm)")
            ax1.set_ylabel("Voltage (V)")
            ax1.set_title("All Sensor Outputs")
            ax1.grid(True, alpha=0.3)
            if len(self.sensors) <= 8:
                ax1.legend()
            
            # Average sensor output
            avg_output = np.mean(sensor_outputs, axis=1)
            ax2 = axes_flat[ax_idx]
            ax_idx += 1
            ax2.plot(positions * 1000, avg_output, 'b-', linewidth=2)
            ax2.set_xlabel("Piston Position (mm)")
            ax2.set_ylabel("Average Voltage (V)")
            ax2.set_title("Average Sensor Output")
            ax2.grid(True, alpha=0.3)
            
            # Create inverse function for position estimation
            # Ensure monotonic for interpolation and handle duplicates
            sorted_indices = np.argsort(avg_output)
            sorted_outputs = avg_output[sorted_indices]
            sorted_positions = positions[sorted_indices] * 1000
            
            # Remove duplicate x values by taking the mean of corresponding y values
            unique_outputs, unique_indices = np.unique(sorted_outputs, return_inverse=True)
            unique_positions = np.array([np.mean(sorted_positions[unique_indices == i]) 
                                       for i in range(len(unique_outputs))])
            
            # Check if we have enough unique points for interpolation
            if len(unique_outputs) < 2:
                print("Warning: Not enough unique sensor outputs for position estimation")
                f_inverse = lambda x: np.full_like(x, unique_positions[0] if len(unique_positions) > 0 else 0)
            else:
                f_inverse = interp1d(unique_outputs, unique_positions, 
                                   kind='linear', fill_value='extrapolate')
            
            # Position accuracy analysis
            if self.get_value("show_accuracy_plot"):
                ax3 = axes_flat[ax_idx]
                ax_idx += 1
                
                # Add noise and calculate position error
                noise_level = 0.01  # 10mV noise
                noisy_voltages = avg_output + np.random.normal(0, noise_level, len(avg_output))
                
                estimated_positions = f_inverse(noisy_voltages)
                position_error = np.abs(estimated_positions - positions * 1000)
                
                ax3.plot(positions * 1000, position_error)
                ax3.axhline(y=self.tank_length * 100, color='r', linestyle='--', label='10% error threshold')
                ax3.set_xlabel("Actual Position (mm)")
                ax3.set_ylabel("Position Error (mm)")
                ax3.set_title("Position Estimation Error")
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                # Error percentage
                ax4 = axes_flat[ax_idx]
                ax_idx += 1
                error_percentage = position_error / (positions * 1000 + 1e-6) * 100
                ax4.plot(positions * 1000, error_percentage)
                ax4.axhline(y=10, color='r', linestyle='--', label='10% target')
                ax4.set_xlabel("Actual Position (mm)")
                ax4.set_ylabel("Error (%)")
                ax4.set_title("Relative Position Error")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                
                # Print accuracy statistics
                print(f"\nAccuracy Analysis:")
                print(f"Average error: {np.mean(position_error):.2f} mm")
                print(f"Max error: {np.max(position_error):.2f} mm")
                print(f"Positions within 10% error: {np.sum(error_percentage < 10) / len(error_percentage) * 100:.1f}%")
            
            # Propellant mass plots
            if self.get_value("show_propellant_mass") and ax_idx < len(axes_flat):
                ax5 = axes_flat[ax_idx]
                ax_idx += 1
                
                # Calculate estimated positions from sensor outputs
                estimated_pos_m = f_inverse(avg_output) / 1000  # Convert to meters
                
                # MON25 properties
                mon25_density = 1440  # kg/m³
                # MMH properties  
                mmh_density = 874  # kg/m³
                
                # Calculate masses based on estimated position
                tank_area = np.pi * (self.tank_id / 2) ** 2
                
                mon25_mass = tank_area * estimated_pos_m * mon25_density
                mmh_mass = tank_area * estimated_pos_m * mmh_density
                
                ax5.plot(positions * 1000, mon25_mass, 'r-', label='MON25', linewidth=2)
                ax5.plot(positions * 1000, mmh_mass, 'b-', label='MMH', linewidth=2)
                ax5.set_xlabel("Actual Position (mm)")
                ax5.set_ylabel("Propellant Mass (kg)")
                ax5.set_title("Estimated Propellant Mass from Sensor Data")
                ax5.grid(True, alpha=0.3)
                ax5.legend()
                
            # Hide unused axes
            for i in range(ax_idx, len(axes_flat)):
                axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def animate_system(self):
        """Create an animated visualization of the system."""
        import matplotlib.animation as animation
        from matplotlib.widgets import Slider
        
        if self.get_value("use_magpy_slider"):
            # Interactive mode with slider
            fig = plt.figure(figsize=(14, 8))
            
            # Create 3D subplot for system visualization
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            
            # Add slider for position control
            ax_slider = plt.axes([0.1, 0.02, 0.35, 0.03])
            position_slider = Slider(ax_slider, 'Position (mm)', 0, self.tank_length * 1000, 
                                   valinit=float(self.get_value("piston_position")))
            
            def update_position(val):
                pos = position_slider.val / 1000  # Convert to meters
                self.update_visualization(ax1, ax2, pos)
                fig.canvas.draw_idle()
            
            position_slider.on_changed(update_position)
            
            # Initial draw
            self.update_visualization(ax1, ax2, float(self.get_value("piston_position")) / 1000)
            plt.show()
            
        else:
            # Animated mode
            fig = plt.figure(figsize=(12, 8))
            
            # Create 3D subplot for system visualization
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            
            # Animation parameters
            num_frames = 50
            positions = np.linspace(0, self.tank_length, num_frames)
        
            def animate(frame):
                ax1.clear()
                ax2.clear()
                pos = positions[frame]
                self.update_visualization(ax1, ax2, pos)
                return ax1, ax2
            
            anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                         interval=100, blit=False)
            
            plt.show()
            return anim
    
    def update_visualization(self, ax1, ax2, pos):
        """Update visualization for given position."""
        # Reset to single position for visualization
        if isinstance(self.magnet, magpy.Collection):
            # Move all magnets in collection
            for mag in self.magnet:
                # Get base position (without path)
                if hasattr(mag.position, 'shape') and len(mag.position.shape) > 1:
                    base_y = mag.position[0][1]
                    base_z = mag.position[0][2]
                else:
                    base_y = mag.position[1] if len(mag.position) > 1 else 0
                    base_z = mag.position[2] if len(mag.position) > 2 else 0
                mag.position = (pos, base_y, base_z)
        else:
            self.magnet.position = (pos, 0, 0)
        
        # Sensors remain at their fixed positions
        sensor_objects = []
        voltages = []
        
        for sensor in self.sensors:
            # Keep sensor at its fixed position
            sensor.sensor_obj.position = sensor.position
            sensor_objects.append(sensor.sensor_obj)
            
            # Calculate field and voltage at this magnet position
            B = self.magnet.getB(sensor.sensor_obj)
            voltage = self.calculate_sensor_output(sensor, B)
            voltages.append(voltage)
        
        # Show 3D system
        magpy.show(self.magnet, *sensor_objects, canvas=ax1, show_path=False)
        
        # Add cylinder representation
        theta = np.linspace(0, 2*np.pi, 50)
        x_cyl = np.array([0, self.tank_length])
        
        # Inner cylinder
        for x in x_cyl:
            y_circle = self.tank_id/2 * np.cos(theta)
            z_circle = self.tank_id/2 * np.sin(theta)
            x_circle = np.full_like(y_circle, x)
            ax1.plot(x_circle, y_circle, z_circle, 'gray', alpha=0.5)
        
        # Cylinder lines
        for i in range(0, len(theta), 6):
            ax1.plot([0, self.tank_length], 
                    [self.tank_id/2 * np.cos(theta[i])] * 2,
                    [self.tank_id/2 * np.sin(theta[i])] * 2,
                    'gray', alpha=0.3)
        
        # Add piston representation
        piston_y = self.tank_id/2 * np.cos(theta)
        piston_z = self.tank_id/2 * np.sin(theta)
        piston_x = np.full_like(piston_y, pos)
        ax1.plot(piston_x, piston_y, piston_z, 'blue', linewidth=3)
        
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("3D System View")
        
        # Plot sensor outputs
        ax2.bar(range(len(voltages)), voltages)
        ax2.set_ylim(0, 5)
        ax2.set_xlabel("Sensor Number")
        ax2.set_ylabel("Voltage (V)")
        ax2.set_title(f"Sensor Outputs at Position: {pos*1000:.1f} mm")
    
    def export_results(self):
        """Export analysis results to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            # Collect system configuration
            config = {
                "system_config": self.get_values(),
                "sensor_database": SENSOR_DATABASE,
                "magnet_database": MAGNET_DATABASE,
                "analysis_timestamp": str(np.datetime64('now'))
            }
            
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"Results exported to {filename}")
            else:
                print("CSV export not yet implemented")
    
    def show_field_visualization(self):
        """Show magnetic field lines and field strength visualization."""
        if not self.magnet:
            print("Please update system first!")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 3D view with field
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_title("3D System View")
        
        # Show magnet and sensors
        sensor_objects = [s.sensor_obj for s in self.sensors]
        magpy.show(self.magnet, *sensor_objects, canvas=ax1)
        
        # Field in XZ plane
        ax2 = fig.add_subplot(222)
        ax2.set_title("Magnetic Field (XZ Plane at Y=0)")
        
        # Create grid for field calculation
        x = np.linspace(-0.1, self.tank_length + 0.1, 50)
        z = np.linspace(-self.tank_id, self.tank_id, 50)
        X, Z = np.meshgrid(x, z)
        
        # Calculate field on grid
        positions = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
        B = self.magnet.getB(positions)
        B_magnitude = np.linalg.norm(B, axis=1).reshape(X.shape) * 1000  # Convert to mT
        
        # Plot field magnitude
        contour = ax2.contourf(X * 1000, Z * 1000, B_magnitude, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax2, label='Field Magnitude (mT)')
        
        # Add field lines
        Bx = B[:, 0].reshape(X.shape)
        Bz = B[:, 2].reshape(X.shape)
        ax2.streamplot(X * 1000, Z * 1000, Bx, Bz, color='white', density=1.5)
        
        # Add tank outline
        ax2.plot([0, self.tank_length * 1000], [self.tank_id/2 * 1000, self.tank_id/2 * 1000], 'r-', linewidth=2)
        ax2.plot([0, self.tank_length * 1000], [-self.tank_id/2 * 1000, -self.tank_id/2 * 1000], 'r-', linewidth=2)
        
        ax2.set_xlabel("X Position (mm)")
        ax2.set_ylabel("Z Position (mm)")
        ax2.set_aspect('equal')
        
        # Field along sensor line
        ax3 = fig.add_subplot(223)
        ax3.set_title("Field Strength at Sensor Height")
        
        x_line = np.linspace(0, self.tank_length, 200)
        z_sensor = self.tank_id / 2 + self.tank_thickness
        positions_line = np.column_stack([x_line, np.zeros(len(x_line)), np.full(len(x_line), z_sensor)])
        B_line = self.magnet.getB(positions_line)
        B_mag_line = np.linalg.norm(B_line, axis=1) * 1000  # mT
        
        ax3.plot(x_line * 1000, B_mag_line)
        ax3.set_xlabel("X Position (mm)")
        ax3.set_ylabel("Field Magnitude (mT)")
        ax3.grid(True, alpha=0.3)
        
        # Field components
        ax4 = fig.add_subplot(224)
        ax4.set_title("Field Components at Sensor Height")
        
        ax4.plot(x_line * 1000, B_line[:, 0] * 1000, 'r-', label='Bx')
        ax4.plot(x_line * 1000, B_line[:, 1] * 1000, 'g-', label='By')
        ax4.plot(x_line * 1000, B_line[:, 2] * 1000, 'b-', label='Bz')
        ax4.set_xlabel("X Position (mm)")
        ax4.set_ylabel("Field Component (mT)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_pyvista_field_lines(self):
        """Show 3D field lines using PyVista following Magpylib docs pattern."""
        try:
            import pyvista as pv
            # Allow empty meshes to prevent errors
            pv.global_theme.allow_empty_mesh = True
        except ImportError:
            print("PyVista not available. Please install it with: pip install pyvista")
            return
        
        if not self.magnet:
            print("Please update system first!")
            return
        
        # Get magnet position
        magnet_pos = float(self.get_value("piston_position")) / 1000  # Convert to m
        
        # Create a 3D grid following the Magpylib docs pattern
        grid_size = 0.3  # 300mm grid size
        grid_spacing = 0.005  # 5mm spacing for good resolution
        grid_points = int(grid_size / grid_spacing) + 1
        
        grid = pv.ImageData(
            dimensions=(grid_points, grid_points, grid_points),
            spacing=(grid_spacing, grid_spacing, grid_spacing),
            origin=(magnet_pos - grid_size/2, -grid_size/2, -grid_size/2)
        )
        
        # Compute B-field on grid and convert to mT like in the docs
        B_field = self.magnet.getB(grid.points) * 1000  # T -> mT
        grid["B"] = B_field
        
        print(f"Grid created: {grid_points}³ points, max |B|: {np.max(np.linalg.norm(B_field, axis=1)):.2f} mT")
        
        # Create seed disc like in the Magpylib docs example
        tank_radius = self.tank_id / 2 / 1000  # Convert to meters
        seed = pv.Disc(
            center=(magnet_pos, 0, 0),
            inner=tank_radius * 0.3,  # Inner radius
            outer=tank_radius * 0.8,  # Outer radius  
            normal=(1, 0, 0),         # Normal along x-axis
            r_res=2,                  # Radial resolution
            c_res=12                  # Circumferential resolution
        )
        
        # Compute streamlines following the Magpylib docs pattern
        try:
            streamlines = grid.streamlines_from_source(
                seed,
                vectors="B",
                max_step_length=0.01,       # Larger steps like docs (0.1 -> 0.01 for our scale)
                max_time=0.02,              # Use max_time like docs
                integration_direction="both",
            )
            print(f"Generated {streamlines.n_points} streamline points")
            
        except Exception as e:
            print(f"Warning: Could not compute streamlines with docs parameters: {e}")
            # Try with more conservative parameters
            try:
                print("Trying with conservative parameters...")
                streamlines = grid.streamlines_from_source(
                    seed,
                    vectors="B", 
                    max_step_length=0.005,
                    max_steps=200,
                    integration_direction="both",
                )
                print(f"Generated {streamlines.n_points} streamline points (conservative)")
            except Exception as e2:
                print(f"Warning: Conservative streamlines also failed: {e2}")
                # Use a simpler grid-based visualization instead
                self.show_field_grid()
                return
        
        # Create plotter
        pl = pv.Plotter()
        
        # Add magnet visualization
        magpy.show(self.magnet, canvas=pl, units_length="m", backend="pyvista")
        
        # Add streamlines following the Magpylib docs pattern
        if streamlines.n_points > 0:
            try:
                # Create tubes with radius like in the docs (0.0002 -> scaled for our system)
                tube_radius = max(0.0005, self.tank_id / 1000 / 200)  # Scale tube radius to system
                streamlines_tubes = streamlines.tube(radius=tube_radius)
                
                if streamlines_tubes.n_points > 0:
                    # Prepare legend parameters like in the docs
                    legend_args = {
                        "title": "B (mT)",
                        "title_font_size": 20,
                        "color": "black",
                        "position_y": 0.25,
                        "vertical": True,
                    }
                    
                    # Add streamlines and legend to scene like in the docs
                    pl.add_mesh(
                        streamlines_tubes,
                        cmap="bwr",  # Use same colormap as docs
                        scalar_bar_args=legend_args,
                    )
                    print(f"✓ Added {streamlines_tubes.n_points} streamline tube points")
                else:
                    # Fallback to lines if tubes are empty
                    pl.add_mesh(streamlines, color="blue", line_width=4, opacity=0.8)
                    print("✓ Added streamlines as lines (tubes were empty)")
                    
            except Exception as e:
                print(f"Warning: Could not create streamline tubes: {e}")
                # Fallback to basic lines
                try:
                    pl.add_mesh(streamlines, color="cyan", line_width=3)
                    print("✓ Added basic streamlines")
                except Exception as e2:
                    print(f"✗ Could not add any streamline visualization: {e2}")
        else:
            print("Warning: No streamlines generated - trying alternative visualization...")
            # Create simple field vectors instead
            self._add_field_vectors_to_plot(pl, magnet_pos)
        
        # Add cylinder outline
        cylinder = pv.Cylinder(
            center=(self.tank_length/2, 0, 0),
            direction=(1, 0, 0),
            radius=self.tank_id/2,
            height=self.tank_length
        )
        pl.add_mesh(cylinder, style="wireframe", color="gray", opacity=0.3)
        
        # Add sensors - they are stationary along the tank
        for i, sensor in enumerate(self.sensors):
            sensor_mesh = pv.Sphere(radius=0.003, center=sensor.position)
            pl.add_mesh(sensor_mesh, color="red", label=f"Sensor {i+1}")
        
        # Set camera and show
        pl.camera.position = (self.tank_length/2 + 0.3, 0.3, 0.3)
        pl.camera.focal_point = (self.tank_length/2, 0, 0)
        pl.show()
    
    def _add_field_vectors_to_plot(self, pl, magnet_pos):
        """Add field vectors as fallback when streamlines fail."""
        try:
            # Create a sparse grid of points around the magnet
            n_points = 8
            radius = self.tank_id / 4
            
            vector_points = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)
                vector_points.extend([
                    [magnet_pos - 0.02, y, z],
                    [magnet_pos, y, z], 
                    [magnet_pos + 0.02, y, z]
                ])
            
            vector_points = np.array(vector_points)
            
            # Calculate field at these points
            B_vectors = self.magnet.getB(vector_points)
            B_magnitude = np.linalg.norm(B_vectors, axis=1)
            
            # Normalize vectors for display
            scale = 0.02  # 20mm vector length
            B_normalized = B_vectors / (B_magnitude[:, np.newaxis] + 1e-10) * scale
            
            # Create arrow meshes
            for i, (point, vector, mag) in enumerate(zip(vector_points, B_normalized, B_magnitude)):
                if mag > 1e-8:  # Only show if field is strong enough
                    arrow = pv.Arrow(start=point, direction=vector, scale=mag*1000)
                    pl.add_mesh(arrow, color="orange", opacity=0.7)
            
            print(f"✓ Added {len(vector_points)} field vector arrows as fallback")
            
        except Exception as e:
            print(f"Warning: Could not create field vectors: {e}")
    
    def show_field_grid(self):
        """Show magnetic field on a grid for detailed analysis."""
        if not self.magnet:
            print("Please update system first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get current magnet position
        magnet_pos = float(self.get_value("piston_position")) / 1000
        
        # Grid parameters
        grid_size = 0.3  # 300mm
        grid_points = 50
        
        # XY plane at z=0
        ax = axes[0, 0]
        x = np.linspace(magnet_pos - grid_size/2, magnet_pos + grid_size/2, grid_points)
        y = np.linspace(-grid_size/2, grid_size/2, grid_points)
        X, Y = np.meshgrid(x, y)
        
        positions = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        B = self.magnet.getB(positions)
        B_mag = np.linalg.norm(B, axis=1).reshape(X.shape) * 1000  # mT
        
        im = ax.contourf(X*1000, Y*1000, B_mag, levels=20, cmap='jet')
        plt.colorbar(im, ax=ax, label='|B| (mT)')
        
        # Add field vectors
        skip = 3
        Bx = B[:, 0].reshape(X.shape)
        By = B[:, 1].reshape(X.shape)
        ax.quiver(X[::skip, ::skip]*1000, Y[::skip, ::skip]*1000, 
                  Bx[::skip, ::skip], By[::skip, ::skip], alpha=0.5)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('B-field in XY plane (z=0)')
        ax.set_aspect('equal')
        
        # XZ plane at y=0
        ax = axes[0, 1]
        z = np.linspace(-grid_size/2, grid_size/2, grid_points)
        X, Z = np.meshgrid(x, z)
        
        positions = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
        B = self.magnet.getB(positions)
        B_mag = np.linalg.norm(B, axis=1).reshape(X.shape) * 1000
        
        im = ax.contourf(X*1000, Z*1000, B_mag, levels=20, cmap='jet')
        plt.colorbar(im, ax=ax, label='|B| (mT)')
        
        # Add tank outline
        ax.axhline(y=self.tank_id/2*1000, color='red', linestyle='--', label='Tank wall')
        ax.axhline(y=-self.tank_id/2*1000, color='red', linestyle='--')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('B-field in XZ plane (y=0)')
        ax.set_aspect('equal')
        
        # Field along sensor line (y=0, z=sensor_radius)
        ax = axes[1, 0]
        x_path = np.linspace(0, self.tank_length, 200)
        sensor_y = 0
        sensor_z = self.tank_id / 2 + self.tank_thickness
        positions = np.column_stack([x_path, np.full_like(x_path, sensor_y), np.full_like(x_path, sensor_z)])
        
        B = self.magnet.getB(positions)
        B_mag = np.linalg.norm(B, axis=1) * 1000
        
        ax.plot(x_path*1000, B_mag, 'b-', linewidth=2, label='Field at sensor height')
        
        # Mark sensor positions
        for i, sensor in enumerate(self.sensors):
            ax.axvline(x=sensor.position[0]*1000, color='red', linestyle=':', alpha=0.5)
            # Calculate field at each sensor
            B_sensor = self.magnet.getB(sensor.sensor_obj)
            B_sensor_mag = np.linalg.norm(B_sensor) * 1000
            ax.plot(sensor.position[0]*1000, B_sensor_mag, 'ro', markersize=8)
        
        ax.axvline(x=magnet_pos*1000, color='red', linestyle='--', label='Magnet position')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('|B| (mT)')
        ax.set_title('Field magnitude along sensor line (sensors marked as red dots)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3D view
        ax = axes[1, 1]
        ax.remove()
        ax = fig.add_subplot(224, projection='3d')
        
        # Show magnet and sensors
        sensor_objs = [s.sensor_obj for s in self.sensors]
        magpy.show(self.magnet, *sensor_objs, canvas=ax)
        
        # Add some field points
        n_points = 10
        test_points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            r = self.tank_id / 2
            test_points.append([magnet_pos, r * np.sin(angle), r * np.cos(angle)])
        
        test_points = np.array(test_points)
        B_test = self.magnet.getB(test_points) * 1000
        
        # Plot field vectors
        for i, (pos, B) in enumerate(zip(test_points, B_test)):
            ax.quiver(pos[0], pos[1], pos[2], B[0], B[1], B[2], 
                     length=0.01, color='blue', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D System View with Field Vectors')
        
        plt.tight_layout()
        plt.show()
    

    
    def calculate_position_accuracy(self, num_sensors: int, sensor_type: str, 
                                  field_strength: float) -> float:
        """Calculate expected position accuracy for given configuration."""
        # Base accuracy depends on sensor type
        sensor_data = SENSOR_DATABASE.get(sensor_type, {})
        
        if sensor_data.get("type") == "digital":
            # Digital sensors have limited resolution
            base_accuracy = 50  # mm
        elif sensor_data.get("type") == "3-axis":
            # 3-axis sensors provide best accuracy
            base_accuracy = 20  # mm
        else:
            # Single-axis magnitude sensors
            base_accuracy = 30  # mm
        
        # Improve with more sensors (sqrt relationship)
        accuracy = base_accuracy / np.sqrt(num_sensors)
        
        # Adjust for field strength (stronger field = better accuracy)
        field_factor = min(field_strength / 10, 2)  # Normalize to 10 mT
        accuracy = accuracy / field_factor
        
        return accuracy
    
    def estimate_propellant_mass(self, piston_position: float) -> float:
        """Estimate propellant mass from piston position."""
        # Calculate volume of propellant
        tank_area = np.pi * (self.tank_id / 2) ** 2
        propellant_volume = tank_area * piston_position  # m³
        
        # Typical propellant density (hydrazine: ~1000 kg/m³)
        propellant_density = 1000  # kg/m³
        
        mass = propellant_volume * propellant_density
        return mass

    def create_default_config(self):
        """Create default configuration with ultra-strong magnets for clear visualization"""
        return {
            'sensor': {
                'model': 'TMAG5170',
                'count': 6,
                'output_type': 'analog',
                'range': '±80mT'
            },
            'magnet': {
                'type': 'single',
                'model': 'N52_Ultra_Strong',  # Much stronger magnet
                'dimensions': [200, 100, 50],  # Much larger: 200mm x 100mm x 50mm
                'magnetization': [0, 0, 2000],  # 2000 mT polarization (ultra-strong)
                'temperature': 20
            },
            'tank': {
                'inner_diameter': 457.2,
                'wall_thickness': 3.94,
                'length': 1000
            },
            'analysis': {
                'piston_positions': list(range(0, 1001, 50)),  # 0 to 1000mm
                'sensor_spacing': 'uniform'
            }
        }


if __name__ == "__main__":
    app = PropellantMeasurementSystem()
    app.run() 