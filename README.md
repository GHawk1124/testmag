# MagUI - Magnetic Field Simulation GUI Library

A simple, one-file library for creating CustomTkinter GUIs with easy plotting integration for Seaborn and Magpylib visualizations.

## Features

- **Easy GUI Creation**: Simple API for creating CustomTkinter windows
- **Multiple Input Widgets**: Entry boxes, dropdowns, sliders, checkboxes
- **Integrated Plotting**: Built-in support for Seaborn and Magpylib visualizations
- **Native Plot Windows**: Opens plots in native matplotlib/magpylib windows with full interactivity
- **Responsive Design**: Scrollable main window with proper layout management
- **Value Management**: Easy access to all widget values

## Installation

The library is already set up in this project with all required dependencies. To run:

```bash
cd magui
python magui.py  # Run the demo
# or
python example.py  # Run the example application
```

## Quick Start

### Basic Usage

```python
from magui import MagUI

# Create the main application
app = MagUI("My Application", (700, 600))

# Add input widgets
app.add_entry("sample_size", "Sample Size:", "100")
app.add_combobox("plot_type", ["histogram", "scatter", "line"], "Plot Type:")
app.add_slider("noise_level", 0.1, 2.0, "Noise Level:", 1.0)
app.add_checkbox("show_kde", "Show KDE", True)

# Add a plot button
def my_plot_function(ax=None, **kwargs):
    # Your plotting code here
    # Widget values are passed as kwargs
    import seaborn as sns
    import numpy as np
    
    sample_size = int(kwargs.get('sample_size', 100))
    data = np.random.randn(sample_size)
    sns.histplot(data, kde=kwargs.get('show_kde', True), ax=ax)

app.add_seaborn_button("Generate Plot", my_plot_function)

# Run the application
app.run()
```

### Magpylib Integration

```python
def create_magnetic_objects(**kwargs):
    import magpylib as magpy
    
    # Use widget values to create magnetic objects
    magnet_strength = float(kwargs.get('magnet_strength', 1000))
    
    magnet = magpy.magnet.Cuboid(
        magnetization=(0, 0, magnet_strength),
        dimension=(1, 1, 1),
        position=(0, 0, 0)
    )
    
    sensor = magpy.Sensor(position=(2, 0, 0))
    return [magnet, sensor]

app.add_magpylib_button("Show Magnetic Field", create_magnetic_objects)
```

## API Reference

### Main Classes

#### `MagUI(title, size)`
Main application window class.

**Parameters:**
- `title` (str): Window title
- `size` (tuple): Window size as (width, height)

### Widget Methods

#### `add_entry(name, label, default, placeholder, **kwargs)`
Add a text entry widget.

#### `add_combobox(name, values, label, default, **kwargs)`
Add a dropdown combobox widget.

#### `add_option_menu(name, values, label, default, **kwargs)`
Add an option menu widget.

#### `add_slider(name, from_, to, label, default, **kwargs)`
Add a slider widget.

#### `add_checkbox(name, text, default, **kwargs)`
Add a checkbox widget.

#### `add_button(text, command, **kwargs)`
Add a regular button.

#### `add_plot_button(text, plot_func, window_title, **kwargs)`
Add a button that opens a plot window.

#### `add_seaborn_button(text, plot_func, window_title, **kwargs)`
Add a button for Seaborn plots.

#### `add_magpylib_button(text, create_objects_func, window_title, **kwargs)`
Add a button for Magpylib visualizations.

### Value Management

#### `get_value(name)`
Get the current value of a named widget.

#### `get_values()`
Get all widget values as a dictionary.

#### `set_value(name, value)`
Set the value of a named widget.

### Utility Methods

#### `run()`
Start the application main loop.

## Examples

### Example 1: Simple Data Visualization

```python
from magui import MagUI
import seaborn as sns
import numpy as np

def create_histogram(ax=None, **kwargs):
    sample_size = int(kwargs.get('sample_size', 100))
    data = np.random.randn(sample_size)
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title(f"Random Data Histogram (n={sample_size})")

app = MagUI("Data Visualizer")
app.add_entry("sample_size", "Sample Size:", "100")
app.add_seaborn_button("Generate Histogram", create_histogram)
app.run()
```

### Example 2: Magnetic Field Simulation

```python
from magui import MagUI
import magpylib as magpy
import numpy as np

def create_magnet_system(**kwargs):
    strength = float(kwargs.get('strength', 1000))
    distance = float(kwargs.get('distance', 2.0))
    
    magnet = magpy.magnet.Cuboid(
        magnetization=(0, 0, strength),
        dimension=(1, 1, 1)
    )
    
    sensor = magpy.Sensor(position=(distance, 0, 0))
    return [magnet, sensor]

app = MagUI("Magnetic Field Simulator")
app.add_slider("strength", 100, 2000, "Magnet Strength:", 1000)
app.add_slider("distance", 0.5, 5.0, "Sensor Distance:", 2.0)
app.add_magpylib_button("Visualize System", create_magnet_system)
app.run()
```

## Native Plot Features

Each plot opens in a native window with:
- **Full interactivity**: Native zoom, pan, and navigation tools
- **Save functionality**: Built-in matplotlib save options
- **Print support**: Native print capabilities
- **Synchronous plotting**: Reliable plot generation on main thread
- **Multiple backends**: Support for matplotlib and magpylib native displays

## Dependencies

- `customtkinter>=5.2.2`
- `magpylib>=5.1.1`
- `matplotlib>=3.10.3`
- `seaborn>=0.13.2`
- `numpy>=2.3.1`

## License

MIT License - See project details for full license information.

## Tips

1. **Widget Values**: All widget values are automatically passed to plot functions as keyword arguments
2. **Native Windows**: Plots open in native matplotlib/magpylib windows for full interactivity
3. **Error Handling**: The library includes basic error handling for common issues
4. **Styling**: Uses CustomTkinter's modern styling with system appearance mode
5. **Scrollable Interface**: Main window is scrollable for applications with many widgets

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed
- **Plot Not Showing**: Check that your plot function accepts the `ax` parameter
- **Widget Values**: Use `app.get_values()` to debug current widget states
- **Magpylib Issues**: Ensure magpylib is properly installed for magnetic field visualizations 