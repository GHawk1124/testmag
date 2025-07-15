import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numba import njit

@njit
def sliding_window_search(meas_volts, calibration_voltages, prev_idx, window):
    start = max(0, prev_idx - window)
    end = min(len(calibration_voltages) - 1, prev_idx + window)
    min_error = float('inf')
    best_idx = prev_idx
    
    for j in range(start, end + 1):
        error = np.sum((meas_volts - calibration_voltages[j])**2)
        if error < min_error:
            min_error = error
            best_idx = j
    
    return best_idx

@njit
def kalman_update(est_pos, k_pos, k_var, q, r):
    k_var += q
    K = k_var / (k_var + r)
    k_pos = k_pos + K * (est_pos - k_pos)
    k_var = (1 - K) * k_var
    return k_pos, k_var

SENSOR_DATABASE = {
    "OMH3040S": {
        "type": "digital",
        "sensitivity": 40.0,  # mT threshold
        "axis": "z",
        "height_offset": 0.001,
        "max_drift": 0.02,  # 2% max drift over full measurement
    },
    "OMH3150S": {
        "type": "magnitude",
        "sensitivity": 35,  # mV/mT (legacy - not used with new transfer function)
        "axis": "xyz",
        "height_offset": 0.001,
        "max_drift": 0.05,  # 5% max drift over full measurement
        "field_range_mT": (-80, 80),  # -800 to +800 Gauss
        "voltage_range_V": (0.5, 4.5),  # Output voltage range
        "transfer_function": lambda B_mT: max(0.5, min(4.5, 0.025 * B_mT + 2.5))  # Linear with clamping
    },
    "TMAG5170": {
        "type": "3-axis",
        "sensitivity": 50.0,  # LSB/mT (12-bit)
        "axis": "xyz",
        "height_offset": 0.001,
        "max_drift": 0.005,  # 0.5% max drift over full measurement
    },
}

MAGNET_DATABASE = {
    "N52": {
        "Br_temp_data": [(-40, 1.48), (20, 1.45), (80, 1.42), (150, 1.38), (200, 1.32)],  # (temp_C, Br_T)
        "BHmax": 80,  # MGOe
    },
    "N42": {
        "Br_temp_data": [(-40, 1.42), (20, 1.38), (80, 1.35), (150, 1.30), (200, 1.22)],  # (temp_C, Br_T)
        "BHmax": 40,  # MGOe
    }
}

for magnet_type, data in MAGNET_DATABASE.items():
    temps, brs = zip(*data["Br_temp_data"])
    data["Br"] = lambda temp, temps=temps, brs=brs: np.interp(temp, temps, brs)
    
    nominal_br = data["Br"](20)  # Br at 20°C
    data["Hc"] = lambda temp, bhmax=data["BHmax"], br_func=data["Br"]: (4 * bhmax * 79577.5) / br_func(temp)  # Convert MGOe to A/m

CYLINDER_RADIUS = (9.6/2.0)+0.155 # in
STROKE = 18 # in
RESOLUTION = 0.05 #in
_RESOLUTION = int(STROKE/RESOLUTION)

# Control whether to run single test with plotting or parameter sweep
RUN_SWEEP = False  # Set to True for parameter sweep, False for single test with plotting

class MagnetRing(magpy.Collection):
    def __init__(self, radius, magradius, magheight, magnet_type, disks=6, orientation='radius', magnetization='z', temperature=20, hc_variation=0.05, **kwargs):
        super().__init__(**kwargs) 
        xs = np.linspace(0, STROKE*0.0254, _RESOLUTION)
        pos = np.array([(t,0,0) for t in xs])
        self.radius = radius
        self.position = pos
        self.magradius = magradius
        self.magheight = magheight
        self.magnet_type = magnet_type
        self.magnet_orientation = orientation
        self.magnetization = magnetization
        self.temperature = temperature
        self.hc_variation = hc_variation
        self._update(disks)

    @property
    def disks(self):
        return self._disks

    @disks.setter
    def disks(self, inp):
        self._update(inp)

    def _update(self, disks):
        self._disks = disks
        ring_radius = self.radius*0.0254
        pos_temp = self.position
        ori_temp = self.orientation
        self.reset_path()
        self.children = []
        self.style.model3d.data.clear()
        
        nominal_hc = MAGNET_DATABASE[self.magnet_type]['Hc'](self.temperature)
            
        for i in range(disks):
            individual_hc = nominal_hc * (1 + np.random.normal(0, self.hc_variation))
            
            if self.magnetization == 'x':
                magnetization = (individual_hc, 0, 0)
            elif self.magnetization == 'y':
                magnetization = (0, individual_hc, 0)
            else:  # 'z'
                magnetization = (0, 0, individual_hc)
                
            child = magpy.magnet.Cylinder(
                magnetization=magnetization,
                dimension=(self.magradius*0.0254, self.magheight*0.0254),
                position=(ring_radius,0,0),
            )
            
            if self.magnet_orientation == 'x':
                child.orientation = R.from_rotvec((0, 0, 90), degrees=True)
            elif self.magnet_orientation == 'y':
                child.orientation = R.from_rotvec((90, 0, 0), degrees=True)
            else:  # 'radius' (radial orientation)
                child.orientation = R.from_rotvec((0, 90, 0), degrees=True)
                
            child.rotate_from_angax(360/disks*i, 'z', anchor=0)
            self.add(child)

        self.position = pos_temp
        self.orientation = ori_temp

        trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(ring_radius-.006, ring_radius+.006, 0.011, 0, 360),
            vert=150,
            opacity=0.2,
        )
        self.style.model3d.add_trace(trace)

        return self
    
def run_piston_length_analysis(cylinder_radius, stroke, resolution, magnet_radius, magnet_height, magnet_disks, num_sensors, sensor_type, magnet_type, sensor_offset, magnet_ring_radius_factor=0.95, magnet_orientation='radius', magnetization_direction='z', temperature=20, kalman_r=1e-6, voltage_noise=0.01, hc_variation=0.05, window=20, plot=True):
    # Set random seed for reproducibility
    np.random.seed(42)
    sensor_positions = np.linspace(sensor_offset*0.0254, (stroke*0.0254)-(sensor_offset*0.0254), num_sensors)
    sensors = [magpy.Sensor(position=(pos, 0, cylinder_radius*0.0254)) for pos in sensor_positions]
    
    # Use the passed resolution parameter to build position grid
    step = resolution * 0.0254  # Convert inches to meters
    actual_positions = np.arange(0, stroke*0.0254 + step/2, step)
    sensor_sensitivity = SENSOR_DATABASE[sensor_type]['sensitivity']

    magnet_ring = MagnetRing(cylinder_radius*magnet_ring_radius_factor, magnet_radius, magnet_height, magnet_type, magnet_disks, orientation=magnet_orientation, magnetization=magnetization_direction, temperature=temperature, hc_variation=hc_variation)
    magnet_ring.rotate_from_angax(angle=90, axis='y')
    
    if plot:  # Only print details when plotting
        print(f"Using {magnet_type} magnet at {temperature}°C:")
        print(f"  Br = {MAGNET_DATABASE[magnet_type]['Br'](temperature):.3f} T")
        print(f"  Hc = {MAGNET_DATABASE[magnet_type]['Hc'](temperature):.0f} A/m")
        print(f"  Hc variation: ±{hc_variation*100:.1f}% (1-sigma)")
        
        print(f"Using {sensor_type} sensor:")
        sensor_info = SENSOR_DATABASE[sensor_type]
        if sensor_info['type'] == 'digital':
            print(f"  Type: Digital threshold sensor")
            print(f"  Threshold: {sensor_info['sensitivity']:.1f} mT")
            print(f"  Output: 5V when B > threshold, 0V otherwise")
        elif sensor_info['type'] == 'magnitude':
            print(f"  Type: Analog magnitude sensor")
            if 'transfer_function' in sensor_info:
                field_min, field_max = sensor_info['field_range_mT']
                volt_min, volt_max = sensor_info['voltage_range_V']
                print(f"  Field range: {field_min:.0f} to {field_max:.0f} mT ({field_min*10:.0f} to {field_max*10:.0f} Gauss)")
                print(f"  Output range: {volt_min:.1f} to {volt_max:.1f} V (clamped)")
                print(f"  Zero field output: {sensor_info['transfer_function'](0):.1f} V")
            else:
                print(f"  Sensitivity: {sensor_info['sensitivity']:.1f} mV/mT")
                print(f"  Output: Proportional voltage")
        elif sensor_info['type'] == '3-axis':
            print(f"  Type: 3-axis digital sensor")
            print(f"  Sensitivity: {sensor_info['sensitivity']:.1f} LSB/mT")
            print(f"  Output: Digital counts (converted to 0-3.3V)")
        
        print(f"  Sensor drift: ±{sensor_info['max_drift']*100:.1f}% max")
        print(f"  Voltage noise: ±{voltage_noise:.3f} V (1-sigma)")
        print(f"  Search window: ±{window} positions")
    
    positions_3d = np.column_stack([actual_positions, np.zeros_like(actual_positions), np.zeros_like(actual_positions)])
    magnet_ring.position = positions_3d
    
    if plot:
        print("Computing magnetic fields...")
    
    # Build CLEAN calibration table (no drift, no noise, no hc_variation, but same temperature)
    calibration_magnet_ring = MagnetRing(cylinder_radius*magnet_ring_radius_factor, magnet_radius, magnet_height, magnet_type, magnet_disks, orientation=magnet_orientation, magnetization=magnetization_direction, temperature=temperature, hc_variation=0.0)  # Clean but same temperature
    calibration_magnet_ring.rotate_from_angax(angle=90, axis='y')
    
    positions_3d = np.column_stack([actual_positions, np.zeros_like(actual_positions), np.zeros_like(actual_positions)])
    calibration_magnet_ring.position = positions_3d
    
    calibration_voltages = np.zeros((len(actual_positions), len(sensors)))
    
    for j, sensor in enumerate(sensors):
        B_fields = calibration_magnet_ring.getB(sensor)
        
        # Generate CLEAN sensor output based on sensor type (no drift, no noise)
        if SENSOR_DATABASE[sensor_type]['type'] == 'digital':
            B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
            threshold = sensor_sensitivity  # mT threshold
            base_voltages = (B_magnitudes > threshold).astype(float) * 5.0  # 5V when above threshold
        
        elif SENSOR_DATABASE[sensor_type]['type'] == 'magnitude':
            sensor_info = SENSOR_DATABASE[sensor_type]
            if 'transfer_function' in sensor_info:
                # Use signed Z-component for Hall effect sensors to get full range
                B_z_signed = B_fields[:, 2] * 1000  # Z-component in mT (signed)
                base_voltages = np.array([sensor_info['transfer_function'](B_z) for B_z in B_z_signed])
            else:
                # Fallback to old linear method using magnitude
                B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
                sensitivity_V_per_mT = sensor_sensitivity / 1000.0  # Convert mV/mT to V/mT
                base_voltages = B_magnitudes * sensitivity_V_per_mT
        
        elif SENSOR_DATABASE[sensor_type]['type'] == '3-axis':
            B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
            sensitivity_LSB_per_mT = sensor_sensitivity
            digital_counts = B_magnitudes * sensitivity_LSB_per_mT
            # Convert to voltage assuming 3.3V reference and 12-bit ADC
            base_voltages = (digital_counts / 4096) * 3.3
        
        calibration_voltages[:, j] = base_voltages  # Clean calibration data
    
    # Now generate NOISY measurement voltages with real-world effects
    magnet_ring.position = positions_3d
    measurement_voltages = np.zeros((len(actual_positions), len(sensors)))
    max_drift = SENSOR_DATABASE[sensor_type]['max_drift']
    
    for j, sensor in enumerate(sensors):
        B_fields = magnet_ring.getB(sensor)  # Uses the magnet ring with temperature, hc_variation, etc.
        
        # Generate sensor output based on sensor type
        if SENSOR_DATABASE[sensor_type]['type'] == 'digital':
            B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
            threshold = sensor_sensitivity  # mT threshold
            base_voltages = (B_magnitudes > threshold).astype(float) * 5.0  # 5V when above threshold
        
        elif SENSOR_DATABASE[sensor_type]['type'] == 'magnitude':
            sensor_info = SENSOR_DATABASE[sensor_type]
            if 'transfer_function' in sensor_info:
                # Use signed Z-component for Hall effect sensors to get full range
                B_z_signed = B_fields[:, 2] * 1000  # Z-component in mT (signed)
                base_voltages = np.array([sensor_info['transfer_function'](B_z) for B_z in B_z_signed])
            else:
                # Fallback to old linear method using magnitude
                B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
                sensitivity_V_per_mT = sensor_sensitivity / 1000.0  # Convert mV/mT to V/mT
                base_voltages = B_magnitudes * sensitivity_V_per_mT
        
        elif SENSOR_DATABASE[sensor_type]['type'] == '3-axis':
            B_magnitudes = np.linalg.norm(B_fields, axis=1) * 1000  # Convert T to mT
            sensitivity_LSB_per_mT = sensor_sensitivity
            digital_counts = B_magnitudes * sensitivity_LSB_per_mT
            # Convert to voltage assuming 3.3V reference and 12-bit ADC
            base_voltages = (digital_counts / 4096) * 3.3
        
        # Apply individual sensor drift (each sensor drifts independently)
        drift_factor = np.random.uniform(-max_drift, max_drift)
        drift_progression = np.linspace(0, drift_factor, len(actual_positions))
        drifted_voltages = base_voltages * (1 + drift_progression)
        
        measurement_voltages[:, j] = drifted_voltages

    estimated_positions = []
    filtered_positions = []
    position_errors = []
    noisy_voltages = []

    prev_idx = 0
    q = (resolution*0.0254)**2
    r = max(kalman_r, voltage_noise**2)  # Adapt to actual noise level
    k_var = 1.0
    k_pos = actual_positions[0]

    for i, meas_volts in enumerate(measurement_voltages):
        if plot:
            print(f"\rProcessing step {i+1}/{len(measurement_voltages)}", end='', flush=True)
        
        # Add measurement noise on top of the already noisy measurement voltages
        noisy_volts = meas_volts + np.random.normal(0, voltage_noise, len(meas_volts))
        noisy_voltages.append(noisy_volts)
        
        if i == 0:
            est_pos = actual_positions[0]
            prev_idx = 0
        else:
            prev_idx = sliding_window_search(noisy_volts, calibration_voltages, prev_idx, window)
            est_pos = actual_positions[prev_idx]
            
            # Debug: Check if we're getting reasonable results
            if plot and i % 500 == 0:  # Print every 500th step
                actual_pos_in = actual_positions[i] / 0.0254
                est_pos_in = est_pos / 0.0254
                print(f"\nStep {i}: Actual={actual_pos_in:.2f}in, Estimated={est_pos_in:.2f}in, Error={abs(actual_pos_in-est_pos_in):.2f}in")
                print(f"  Meas volts: [{noisy_volts[0]:.3f}, {noisy_volts[1]:.3f}, {noisy_volts[2]:.3f}...]")
                print(f"  Cal volts at actual pos: [{calibration_voltages[i][0]:.3f}, {calibration_voltages[i][1]:.3f}, {calibration_voltages[i][2]:.3f}...]")
                print(f"  Cal volts at est pos: [{calibration_voltages[prev_idx][0]:.3f}, {calibration_voltages[prev_idx][1]:.3f}, {calibration_voltages[prev_idx][2]:.3f}...]")
            
            # Check for tracking failure
            if plot and abs(est_pos - actual_positions[i]) > 0.1:  # 100mm threshold
                print(f"\nWarning: Possible tracking failure at step {i+1}")
                print(f"  Estimated: {est_pos/0.0254:.1f} in, Actual: {actual_positions[i]/0.0254:.1f} in")
        estimated_positions.append(est_pos)

        k_pos, k_var = kalman_update(est_pos, k_pos, k_var, q, r)

        filtered_positions.append(k_pos)
        position_errors.append(abs(k_pos - actual_positions[i]))

    estimated_positions = np.array(estimated_positions)
    filtered_positions = np.array(filtered_positions)
    position_errors = np.array(position_errors)
    noisy_voltages = np.array(noisy_voltages)
    all_voltages = noisy_voltages
    
    if plot:
        print(f"\nMean position error: {np.mean(position_errors)/0.0254:.3f} inches")
        print(f"Max position error: {np.max(position_errors)/0.0254:.3f} inches")
        print(f"Position accuracy (1-sigma): {np.std(position_errors)/0.0254:.3f} inches")
    
    if plot:
        plot_frames = 100
        downsample_indices = np.linspace(0, len(actual_positions)-1, plot_frames, dtype=int)
        plot_positions = actual_positions[downsample_indices]
        plot_positions_3d = np.column_stack([plot_positions, np.zeros_like(plot_positions), np.zeros_like(plot_positions)])
        magnet_ring.position = plot_positions_3d
        
        tank_trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(cylinder_radius*0.0254, cylinder_radius*0.0254+0.005, stroke*0.0254, 0, 360),
            vert=150,
            opacity=0.1,
        )
        tank_collection = magpy.Collection()
        tank_collection.style.model3d.add_trace(tank_trace)
        tank_collection.position = (stroke*0.0254/2, 0, 0)
        tank_collection.rotate_from_angax(angle=90, axis='y')
        
        magpy.show(dict(objects=[magnet_ring]+sensors+[tank_collection], output=["B"], col=1, row=1),
                   dict(objects=[magnet_ring]+sensors+[tank_collection], output="model3d", col=2, row=1),
                   backend="plotly",
                   animation=True)
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Position Error (inches)', 'Sensor Voltages vs Position', 
                                         'Position Error (% of stroke)', 'Estimation Accuracy'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}]])
        
        fig.add_trace(go.Scatter(x=actual_positions/0.0254, y=position_errors/0.0254, 
                                name='Position Error', mode='lines'), row=1, col=1)
        
        for i in range(num_sensors):
            fig.add_trace(go.Scatter(x=actual_positions/0.0254, y=all_voltages[:, i], 
                                    name=f'Sensor {i+1}', mode='lines'), row=1, col=2)
        
        percentage_error = (position_errors / (stroke*0.0254)) * 100
        fig.add_trace(go.Scatter(x=actual_positions/0.0254, y=percentage_error, 
                                name='Error %', mode='lines'), row=2, col=1)
        fig.add_hline(y=5, line_dash="dash", line_color="red", 
                     annotation_text="5% Threshold", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=actual_positions[1:]/0.0254, y=filtered_positions[1:]/0.0254,
                                name='Estimated', mode='lines'), row=2, col=2)
        fig.add_trace(go.Scatter(x=actual_positions[1:]/0.0254, y=actual_positions[1:]/0.0254,
                                name='Actual', mode='lines', line=dict(dash='dash')), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Piston Position Analysis Dashboard")
        fig.show()
    
    return position_errors

if __name__ == "__main__":
    import pandas as pd
    import time
    from datetime import datetime
    
    if not RUN_SWEEP:
        position_errors_low = run_piston_length_analysis(
            cylinder_radius=CYLINDER_RADIUS,
            stroke=STROKE,
            resolution=0.001,
            magnet_radius=0.25,
            magnet_height=0.125,
            magnet_disks=6,
            num_sensors=10,
            sensor_type="OMH3150S",
            magnet_type="N52",
            sensor_offset=1.0,
            magnet_ring_radius_factor=0.875,
            magnet_orientation='x',
            magnetization_direction='z',
            temperature=-10,
            kalman_r=1e-4,
            voltage_noise=0.01,
            hc_variation=0.05,
            window=200,
            plot=True
        )
    else:
        # Parameter sweep configuration
        parameters = {
            'resolution': [0.05],
            'magnet_radius': [0.125, 0.25, 0.5],
            'magnet_height': [0.125, 0.25],
            'magnet_disks': [6],
            'magnet_ring_radius_factor': [0.85, 0.9, 0.95],
            'num_sensors': [6, 8, 10, 12],
            'sensor_type': ["OMH3150S"],
            'magnet_type': ["N52", "N42"],
            'sensor_offset': [0.0, 0.5, 1.0],
            'magnet_orientation': ['radius'],
            'magnetization_direction': ['z'],
            'temperature': [-10, 0, 10],
            'kalman_r': [1e-6, 1e-4],
            'voltage_noise': [0.05],
            'hc_variation': [0.1],
            'window': [20]
        }
        
        # Calculate total combinations
        total_combinations = 1
        for param_list in parameters.values():
            total_combinations *= len(param_list)
        
        print(f"Starting comprehensive parameter sweep...")
        print(f"Total combinations: {total_combinations:,}")
        
        # Results storage
        results_data = []
        combination_count = 0
        start_time = time.time()
    
        # Nested loops for all parameter combinations
        for resolution in parameters['resolution']:
            for mag_radius in parameters['magnet_radius']:
                for mag_height in parameters['magnet_height']:
                    for mag_disks in parameters['magnet_disks']:
                        for mag_ring_radius_factor in parameters['magnet_ring_radius_factor']:
                            for num_sens in parameters['num_sensors']:
                                for sens_type in parameters['sensor_type']:
                                    for mag_type in parameters['magnet_type']:
                                        for sens_offset in parameters['sensor_offset']:
                                            for mag_orient in parameters['magnet_orientation']:
                                                for mag_direction in parameters['magnetization_direction']:
                                                    for temp in parameters['temperature']:
                                                        for kalman_r in parameters['kalman_r']:
                                                            for v_noise in parameters['voltage_noise']:
                                                                for hc_var in parameters['hc_variation']:
                                                                    for window in parameters['window']:
                                                                        combination_count += 1
                                                                        
                                                                        # Progress update
                                                                        if combination_count % 100 == 0:
                                                                            elapsed = time.time() - start_time
                                                                            rate = combination_count / elapsed
                                                                            eta = (total_combinations - combination_count) / rate
                                                                            #print(f"Progress: {combination_count:,}/{total_combinations:,} ({combination_count/total_combinations*100:.1f}%) - ETA: {eta/60:.1f} min")
                                                                        
                                                                        # Skip invalid combinations
                                                                        if sens_offset >= STROKE * 0.8:
                                                                            continue
                                                                        
                                                                        try:
                                                                            # Run analysis
                                                                            position_errors = run_piston_length_analysis(
                                                                                cylinder_radius=CYLINDER_RADIUS,
                                                                                stroke=STROKE,
                                                                                resolution=resolution,
                                                                                magnet_radius=mag_radius,
                                                                                magnet_height=mag_height,
                                                                                magnet_disks=mag_disks,
                                                                                num_sensors=num_sens,
                                                                                sensor_type=sens_type,
                                                                                magnet_type=mag_type,
                                                                                sensor_offset=sens_offset,
                                                                                magnet_ring_radius_factor=mag_ring_radius_factor,
                                                                                magnet_orientation=mag_orient,
                                                                                magnetization_direction=mag_direction,
                                                                                temperature=temp,
                                                                                kalman_r=kalman_r,
                                                                                voltage_noise=v_noise,
                                                                                hc_variation=hc_var,
                                                                                window=window,
                                                                                plot=False
                                                                            )
                                                                            
                                                                            # Store results
                                                                            result = {
                                                                                'combination': combination_count,
                                                                                'resolution': resolution,
                                                                                'magnet_radius': mag_radius,
                                                                                'magnet_height': mag_height,
                                                                                'magnet_disks': mag_disks,
                                                                                'magnet_ring_radius_factor': mag_ring_radius_factor,
                                                                                'num_sensors': num_sens,
                                                                                'sensor_type': sens_type,
                                                                                'magnet_type': mag_type,
                                                                                'sensor_offset': sens_offset,
                                                                                'magnet_orientation': mag_orient,
                                                                                'magnetization_direction': mag_direction,
                                                                                'temperature': temp,
                                                                                'kalman_r': kalman_r,
                                                                                'voltage_noise': v_noise,
                                                                                'hc_variation': hc_var,
                                                                                'window': window,
                                                                                'mean_error_mm': float(np.mean(position_errors) * 1000),
                                                                                'max_error_mm': float(np.max(position_errors) * 1000),
                                                                                'std_error_mm': float(np.std(position_errors) * 1000),
                                                                                'mean_error_percent': float(np.mean(position_errors) / (STROKE * 0.0254) * 100),
                                                                                'max_error_percent': float(np.max(position_errors) / (STROKE * 0.0254) * 100),
                                                                                'positions_within_5_percent': float(np.sum(position_errors / (STROKE * 0.0254) < 0.05) / len(position_errors) * 100)
                                                                            }
                                                                            results_data.append(result)
                                                                            
                                                                            # Print best results so far
                                                                            # if result['mean_error_percent'] < 2.0:
                                                                            #     print(f"  Good result: {result['mean_error_percent']:.2f}% avg error")
                                                                            #     print(f"    Config: {num_sens} sensors, {sens_type}, {mag_type}, {temp}°C, noise={v_noise:.3f}")
                                                                        
                                                                        except Exception as e:
                                                                            print(f"  Error in combination {combination_count}: {e}")
                                                                            continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save results to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"parameter_sweep_results_{timestamp}.csv"
        pickle_filename = f"parameter_sweep_results_{timestamp}.pkl"
        
        results_df.to_csv(csv_filename, index=False)
        results_df.to_pickle(pickle_filename)
        
        print(f"\nParameter sweep completed!")
        print(f"Total combinations tested: {len(results_df):,}")
        print(f"Results saved to: {csv_filename} and {pickle_filename}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        
        # Print summary statistics
        if not results_df.empty:
            print(f"\nSummary:")
            print(f"Best mean error: {results_df['mean_error_percent'].min():.2f}%")
            print(f"Worst mean error: {results_df['mean_error_percent'].max():.2f}%")
            print(f"Average mean error: {results_df['mean_error_percent'].mean():.2f}%")
            print(f"Configurations with <5% error: {len(results_df[results_df['mean_error_percent'] < 5.0])}/{len(results_df)}")
            
            # Find best configuration
            best_row = results_df.loc[results_df['mean_error_percent'].idxmin()]
            print(f"\nBest configuration:")
            param_cols = ['resolution', 'magnet_radius', 'magnet_height', 'magnet_disks', 'magnet_ring_radius_factor', 
                         'num_sensors', 'sensor_type', 'magnet_type', 'sensor_offset', 
                         'magnet_orientation', 'magnetization_direction', 'temperature', 'kalman_r', 
                         'voltage_noise', 'hc_variation', 'window']
            for param in param_cols:
                print(f"  {param}: {best_row[param]}")
            print(f"  Mean error: {best_row['mean_error_percent']:.2f}%")
            
            # Additional pandas analysis
            print(f"\nTop 5 sensor types by mean error:")
            sensor_analysis = results_df.groupby('sensor_type')['mean_error_percent'].agg(['mean', 'min', 'count']).sort_values('mean')
            print(sensor_analysis)
            
            print(f"\nTop 5 magnet types by mean error:")
            magnet_analysis = results_df.groupby('magnet_type')['mean_error_percent'].agg(['mean', 'min', 'count']).sort_values('mean')
            print(magnet_analysis)
