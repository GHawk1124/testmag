#!/usr/bin/env python3
"""
Example usage of the MagUI library.

This example shows how to create a CustomTkinter GUI with input widgets
and buttons that generate plots using Seaborn and Magpylib.
"""

import numpy as np
import seaborn as sns
from magui import MagUI


def custom_seaborn_plot(ax=None, sample_size=100, plot_type="histogram", 
                       noise_level=1.0, show_kde=True, **kwargs):
    """Custom seaborn plotting function that uses widget values."""
    # Convert string sample_size to int
    try:
        sample_size = int(sample_size)
    except (ValueError, TypeError):
        sample_size = 100
    
    # Generate sample data with specified noise level
    np.random.seed(42)
    if plot_type == "histogram":
        data = np.random.normal(0, noise_level, sample_size)
        sns.histplot(data, kde=show_kde, ax=ax)
        ax.set_title(f"Histogram (n={sample_size}, noise={noise_level:.1f})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        
    elif plot_type == "scatter":
        x = np.random.normal(0, noise_level, sample_size)
        y = x * 2 + np.random.normal(0, noise_level * 0.5, sample_size)
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.set_title(f"Scatter Plot (n={sample_size}, noise={noise_level:.1f})")
        ax.set_xlabel("X Value")
        ax.set_ylabel("Y Value")
        
    elif plot_type == "line":
        x = np.linspace(0, 10, sample_size)
        y = np.sin(x) + np.random.normal(0, noise_level * 0.1, sample_size)
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_title(f"Line Plot (n={sample_size}, noise={noise_level:.1f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")


def create_magnetic_system(magnet_strength=1000, magnet_size=1.0, 
                          sensor_distance=2.0, **kwargs):
    """Create magpylib objects based on widget values."""
    try:
        import magpylib as magpy
        
        # Convert string values to float
        try:
            magnet_strength = float(magnet_strength)
            magnet_size = float(magnet_size)
            sensor_distance = float(sensor_distance)
        except (ValueError, TypeError):
            magnet_strength = 1000
            magnet_size = 1.0
            sensor_distance = 2.0
        
        # Create a cubic magnet
        magnet = magpy.magnet.Cuboid(
            magnetization=(0, 0, magnet_strength), 
            dimension=(magnet_size, magnet_size, magnet_size),
            position=(0, 0, 0)
        )
        
        # Create a sensor at specified distance
        sensor = magpy.Sensor(position=(sensor_distance, 0, 0))
        
        # Create a path for field calculation
        path = magpy.Sensor(position=np.linspace((-3, 0, 0), (3, 0, 0), 20))
        
        return [magnet, sensor, path]
        
    except ImportError:
        print("Magpylib not available")
        return []


def create_advanced_app():
    """Create a more advanced application with multiple widgets."""
    app = MagUI("Advanced MagUI Example", (800, 700))
    
    # Add header
    app.add_label("🔬 Advanced MagUI Example", font=("Arial", 18, "bold"))
    app.add_label("Configure parameters and generate visualizations", font=("Arial", 12))
    
    # Data generation parameters
    app.add_label("\n📊 Data Parameters", font=("Arial", 14, "bold"))
    app.add_entry("sample_size", "Sample Size:", "200", "Number of data points")
    app.add_slider("noise_level", 0.1, 3.0, "Noise Level:", 1.0)
    app.add_combobox("plot_type", ["histogram", "scatter", "line"], "Plot Type:", "histogram")
    app.add_checkbox("show_kde", "Show KDE (for histogram)", True)
    
    # Magnetic system parameters
    app.add_label("\n🧲 Magnetic System Parameters", font=("Arial", 14, "bold"))
    app.add_entry("magnet_strength", "Magnetization (mT):", "1000", "Magnetic field strength")
    app.add_slider("magnet_size", 0.5, 3.0, "Magnet Size:", 1.0)
    app.add_slider("sensor_distance", 1.0, 5.0, "Sensor Distance:", 2.0)
    
    # Action buttons
    app.add_label("\n🎯 Actions", font=("Arial", 14, "bold"))
    app.add_seaborn_button("📈 Generate Custom Seaborn Plot", 
                          custom_seaborn_plot, "Custom Data Visualization")
    app.add_magpylib_button("🧲 Visualize Magnetic System", 
                           create_magnetic_system, "Magnetic Field Visualization")
    
    # Utility button
    app.add_button("📋 Print Current Values", lambda: print("Values:", app.get_values()))
    
    return app


def simple_example():
    """Create a simple example application."""
    app = MagUI("Simple MagUI Example", (600, 400))
    
    app.add_label("🚀 Simple Example", font=("Arial", 16, "bold"))
    
    # Add a few widgets
    app.add_entry("data_points", "Number of Points:", "50")
    app.add_combobox("chart_style", ["seaborn", "classic", "whitegrid"], "Style:", "seaborn")
    
    # Simple plotting function
    def simple_plot(ax=None, data_points="50", chart_style="seaborn", **kwargs):
        try:
            n = int(data_points)
        except:
            n = 50
            
        # Set style
        sns.set_style(chart_style if chart_style in ["whitegrid", "darkgrid", "white", "dark", "ticks"] else "whitegrid")
        
        # Generate and plot data
        x = np.linspace(0, 2*np.pi, n)
        y = np.sin(x) + 0.1 * np.random.randn(n)
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_title(f"Sine Wave with Noise (n={n})")
    
    app.add_seaborn_button("Generate Plot", simple_plot)
    
    return app


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Simple example
    # app = simple_example()
    
    # Advanced example
    app = create_advanced_app()
    
    # Run the application
    app.run() 