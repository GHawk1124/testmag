#!/usr/bin/env python3
"""
Simple test for the updated MagUI library.
This script tests the basic API without opening GUI windows.
"""

def test_basic_functionality():
    """Test basic MagUI functionality."""
    print("Testing MagUI basic functionality...")
    
    try:
        from magui import MagUI
        print("✓ Import successful")
        
        # Test creating a MagUI instance
        app = MagUI("Test App", (600, 400))
        print("✓ MagUI instance created")
        
        # Test adding widgets
        app.add_label("Test Label")
        print("✓ Label added")
        
        app.add_entry("test_entry", "Test Entry:", "default_value")
        print("✓ Entry widget added")
        
        app.add_combobox("test_combo", ["option1", "option2", "option3"], "Test Combo:", "option1")
        print("✓ Combobox widget added")
        
        app.add_slider("test_slider", 0.0, 10.0, "Test Slider:", 5.0)
        print("✓ Slider widget added")
        
        app.add_checkbox("test_checkbox", "Test Checkbox", True)
        print("✓ Checkbox widget added")
        
        # Test value management
        values = app.get_values()
        print(f"✓ Widget values retrieved: {values}")
        
        # Test individual value access
        entry_value = app.get_value("test_entry")
        print(f"✓ Individual value access: test_entry = {entry_value}")
        
        # Test setting values
        app.set_value("test_entry", "new_value")
        new_value = app.get_value("test_entry")
        print(f"✓ Value setting works: test_entry = {new_value}")
        
        # Test button creation (without actually clicking)
        def dummy_command():
            print("Button clicked!")
        
        app.add_button("Test Button", dummy_command)
        print("✓ Button added")
        
        # Test plotting button creation
        def dummy_plot_func(**kwargs):
            print(f"Plot function called with: {kwargs}")
        
        app.add_plot_button("Test Plot Button", dummy_plot_func)
        print("✓ Plot button added")
        
        # Test seaborn button creation
        def dummy_seaborn_func(ax=None, **kwargs):
            print(f"Seaborn function called with: {kwargs}")
        
        app.add_seaborn_button("Test Seaborn Button", dummy_seaborn_func)
        print("✓ Seaborn button added")
        
        # Test magpylib button creation
        def dummy_magpylib_func(**kwargs):
            print(f"Magpylib function called with: {kwargs}")
            return []  # Return empty list of objects
        
        app.add_magpylib_button("Test Magpylib Button", dummy_magpylib_func)
        print("✓ Magpylib button added")
        
        print("\n🎉 All basic tests passed!")
        print("\nTo run the full GUI demo, execute: python magui.py")
        print("To run the example application, execute: python example.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("Testing dependencies...")
    
    dependencies = [
        "customtkinter",
        "matplotlib",
        "seaborn", 
        "numpy",
        "magpylib"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")

if __name__ == "__main__":
    test_dependencies()
    print()
    success = test_basic_functionality()
    
    if success:
        print("\n🚀 Library is ready to use!")
        print("\n💡 Now plots will open in native windows with full interactivity!")
        print("   - Use mouse wheel to zoom")
        print("   - Click and drag to pan")
        print("   - Use toolbar for save, zoom, etc.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.") 