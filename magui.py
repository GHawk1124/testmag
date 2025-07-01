"""
MagUI - A simple library for creating CustomTkinter GUIs with easy plotting integration.

This library provides a simple interface for creating CustomTkinter windows with:
- Easy addition of input widgets (Entry, ComboBox, OptionMenu)
- Simple button creation that can spawn new windows with plots
- Integration with Seaborn and Magpylib for data visualization
"""

import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union


class MagUI(ctk.CTk):
    """Main application window with easy widget addition and plotting capabilities."""
    
    def __init__(self, title: str = "MagUI Application", size: tuple = (600, 500)):
        super().__init__()
        
        # Configure window
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        
        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        # Storage for widgets and their values
        self.widgets: Dict[str, ctk.CTkBaseClass] = {}
        self.widget_vars: Dict[str, Union[ctk.StringVar, ctk.IntVar, ctk.DoubleVar]] = {}
        
        # Main container with scrollable frame
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current row for grid layout
        self.current_row = 0
    
    def add_label(self, text: str, **kwargs) -> ctk.CTkLabel:
        """Add a label to the window."""
        label = ctk.CTkLabel(self.main_frame, text=text, **kwargs)
        label.grid(row=self.current_row, column=0, columnspan=2, 
                  sticky="w", padx=10, pady=5)
        self.current_row += 1
        return label
    
    def add_entry(self, name: str, label: str = None, default: str = "", 
                 placeholder: str = "", **kwargs) -> ctk.CTkEntry:
        """Add an entry widget with optional label."""
        if label:
            label_widget = ctk.CTkLabel(self.main_frame, text=label)
            label_widget.grid(row=self.current_row, column=0, sticky="w", padx=10, pady=5)
        
        var = ctk.StringVar(value=default)
        entry = ctk.CTkEntry(self.main_frame, textvariable=var, 
                           placeholder_text=placeholder, **kwargs)
        entry.grid(row=self.current_row, column=1, sticky="ew", padx=10, pady=5)
        
        self.widgets[name] = entry
        self.widget_vars[name] = var
        self.current_row += 1
        
        # Configure column weight for responsive design
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        return entry
    
    def add_combobox(self, name: str, values: List[str], label: str = None, 
                    default: str = "", **kwargs) -> ctk.CTkComboBox:
        """Add a combobox (dropdown) widget."""
        if label:
            label_widget = ctk.CTkLabel(self.main_frame, text=label)
            label_widget.grid(row=self.current_row, column=0, sticky="w", padx=10, pady=5)
        
        var = ctk.StringVar(value=default if default else (values[0] if values else ""))
        combobox = ctk.CTkComboBox(self.main_frame, values=values, variable=var, **kwargs)
        combobox.grid(row=self.current_row, column=1, sticky="ew", padx=10, pady=5)
        
        self.widgets[name] = combobox
        self.widget_vars[name] = var
        self.current_row += 1
        
        return combobox
    
    def add_option_menu(self, name: str, values: List[str], label: str = None, 
                       default: str = "", **kwargs) -> ctk.CTkOptionMenu:
        """Add an option menu (dropdown) widget."""
        if label:
            label_widget = ctk.CTkLabel(self.main_frame, text=label)
            label_widget.grid(row=self.current_row, column=0, sticky="w", padx=10, pady=5)
        
        var = ctk.StringVar(value=default if default else (values[0] if values else ""))
        option_menu = ctk.CTkOptionMenu(self.main_frame, values=values, variable=var, **kwargs)
        option_menu.grid(row=self.current_row, column=1, sticky="ew", padx=10, pady=5)
        
        self.widgets[name] = option_menu
        self.widget_vars[name] = var
        self.current_row += 1
        
        return option_menu
    
    def add_slider(self, name: str, from_: float, to: float, label: str = None, 
                  default: float = None, **kwargs) -> ctk.CTkSlider:
        """Add a slider widget."""
        if default is None:
            default = (from_ + to) / 2
            
        if label:
            label_widget = ctk.CTkLabel(self.main_frame, text=label)
            label_widget.grid(row=self.current_row, column=0, sticky="w", padx=10, pady=5)
        
        var = ctk.DoubleVar(value=default)
        slider = ctk.CTkSlider(self.main_frame, from_=from_, to=to, variable=var, **kwargs)
        slider.grid(row=self.current_row, column=1, sticky="ew", padx=10, pady=5)
        
        self.widgets[name] = slider
        self.widget_vars[name] = var
        self.current_row += 1
        
        return slider
    
    def add_checkbox(self, name: str, text: str, default: bool = False, **kwargs) -> ctk.CTkCheckBox:
        """Add a checkbox widget."""
        var = ctk.BooleanVar(value=default)
        checkbox = ctk.CTkCheckBox(self.main_frame, text=text, variable=var, **kwargs)
        checkbox.grid(row=self.current_row, column=0, columnspan=2, 
                     sticky="w", padx=10, pady=5)
        
        self.widgets[name] = checkbox
        self.widget_vars[name] = var
        self.current_row += 1
        
        return checkbox
    
    def add_button(self, text: str, command: Callable, **kwargs) -> ctk.CTkButton:
        """Add a button widget."""
        button = ctk.CTkButton(self.main_frame, text=text, command=command, **kwargs)
        button.grid(row=self.current_row, column=0, columnspan=2, 
                   sticky="ew", padx=10, pady=10)
        self.current_row += 1
        
        return button
    
    def add_plot_button(self, text: str, plot_func: Callable, 
                       window_title: str = "Plot", **kwargs) -> ctk.CTkButton:
        """Add a button that opens a native matplotlib plot window."""
        def open_plot():
            try:
                # Get current widget values
                values = self.get_values()
                
                # Create new matplotlib figure
                plt.figure(figsize=(10, 6))
                plt.suptitle(window_title)
                
                # Call the plotting function
                plot_func(**values)
                
                # Show the plot in native window
                plt.show()
            except Exception as e:
                print(f"Error in plotting function: {e}")
        
        return self.add_button(text, open_plot, **kwargs)
    
    def add_seaborn_button(self, text: str, plot_func: Callable, 
                          window_title: str = "Seaborn Plot", **kwargs) -> ctk.CTkButton:
        """Add a button that opens a seaborn plot in a native window."""
        def create_seaborn_plot():
            try:
                # Get current widget values
                values = self.get_values()
                
                # Set seaborn style
                sns.set_style("whitegrid")
                
                # Create new figure
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.suptitle(window_title)
                
                # Call the plotting function with the axes
                plot_func(ax=ax, **values)
                
                # Show the plot
                plt.show()
            except Exception as e:
                print(f"Error in seaborn plotting function: {e}")
        
        return self.add_button(text, create_seaborn_plot, **kwargs)
    
    def add_magpylib_button(self, text: str, create_objects_func: Callable, 
                           window_title: str = "Magpylib Visualization", **kwargs) -> ctk.CTkButton:
        """Add a button that opens a magpylib visualization in a native window."""
        def create_magpylib_plot():
            try:
                import magpylib as magpy
                
                # Get current widget values
                values = self.get_values()
                
                # Create magpylib objects using the provided function
                objects = create_objects_func(**values)
                
                if objects:  # Only show if we have objects
                    # Use magpylib's native show function (opens in separate window)
                    magpy.show(*objects, backend='matplotlib')
                else:
                    print("No magpylib objects to display")
                
            except ImportError:
                print("Magpylib not available. Please install it to use magpylib plotting.")
            except Exception as e:
                print(f"Error creating magpylib visualization: {e}")
        
        return self.add_button(text, create_magpylib_plot, **kwargs)
    
    def get_value(self, name: str) -> Any:
        """Get the current value of a named widget."""
        if name in self.widget_vars:
            return self.widget_vars[name].get()
        return None
    
    def get_values(self) -> Dict[str, Any]:
        """Get all current widget values as a dictionary."""
        return {name: var.get() for name, var in self.widget_vars.items()}
    
    def set_value(self, name: str, value: Any):
        """Set the value of a named widget."""
        if name in self.widget_vars:
            self.widget_vars[name].set(value)
    

    
    def run(self):
        """Start the application main loop."""
        self.mainloop()


# Convenience functions for quick setup
def create_demo_seaborn_plot(ax=None, **kwargs):
    """Demo seaborn plot function."""
    # Generate sample data
    np.random.seed(42)
    
    # Get parameters from kwargs with defaults
    sample_size = int(kwargs.get('sample_size', 100))
    noise_level = float(kwargs.get('noise_level', 1.0))
    show_kde = kwargs.get('show_kde', True)
    
    # Generate data
    data = np.random.normal(0, noise_level, sample_size)
    
    # Create plot
    sns.histplot(data, kde=show_kde, ax=ax)
    ax.set_title(f"Demo Seaborn Plot (n={sample_size}, noise={noise_level:.1f})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")


def create_demo_magpylib_objects(**kwargs):
    """Demo magpylib objects creation function."""
    try:
        import magpylib as magpy
        
        # Create a simple magnet and sensor
        magnet = magpy.magnet.Cuboid(
            magnetization=(0, 0, 1000), 
            dimension=(1, 1, 1),
            position=(0, 0, 0)
        )
        
        sensor = magpy.Sensor(position=(2, 0, 0))
        
        return [magnet, sensor]
    except ImportError:
        print("Magpylib not available for demo")
        return []


# Quick start function
def quick_demo():
    """Create a quick demo application."""
    app = MagUI("MagUI Demo", (700, 600))
    
    app.add_label("📊 MagUI Demo Application", font=("Arial", 16, "bold"))
    app.add_label("Add some parameters and click the buttons to see plots!")
    
    # Add some input widgets
    app.add_entry("sample_size", "Sample Size:", "100", "Enter number of samples")
    app.add_combobox("plot_type", ["histogram", "scatter", "line"], "Plot Type:", "histogram")
    app.add_slider("noise_level", 0.1, 2.0, "Noise Level:", 1.0)
    app.add_checkbox("show_kde", "Show KDE", True)
    
    # Add plot buttons
    app.add_seaborn_button("📈 Generate Seaborn Plot", create_demo_seaborn_plot, "Seaborn Demo")
    app.add_magpylib_button("🧲 Show Magpylib Demo", create_demo_magpylib_objects, "Magpylib Demo")
    
    # Add utility button
    app.add_button("📋 Print Current Values", lambda: print("Values:", app.get_values()))
    
    return app


if __name__ == "__main__":
    # Run the demo
    demo_app = quick_demo()
    demo_app.run() 