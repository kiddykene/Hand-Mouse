import json
import os

CONFIG_FILE = "config.json"


def get_default_config():
    return {
        "SMOOTH_ALPHA": 0.06,
        "FAST_MOVEMENT_ALPHA": 0.45,
        "FAST_MOVEMENT_THRESHOLD": 0.1,
        "fov_factor": 0.31,
        "primary_color": "#626270",
        "rotation_factor": 0.2,
        "raw_mode_threshold": 0.35,
        "raw_mode_switch_time": 1.7,
        "click_offset_deg": 110.0,
        "click_offset_amount": 0.05,
        "raw_mode_sensitivity_x": 0.00094,
        "raw_mode_sensitivity_y": 0.0011,
        "raw_mode_deadzone": 0.09,
        "interframe_lerp": 0.75,
        "pointer_max_substeps": 12,
        "movements_per_second": 10,
        "capture_key": "f8",
        "frame_interval_ms": 20,
        "click_timer": 0.1,
        "r_thresholds": {
            "Right Hand - Thumb": 0.66, "Left Hand - Pinky": 0.48, "Left Hand - Ring": 0.55,
            "Left Hand - Middle": 0.56, "Left Hand - Index": 0.55, "Left Hand - Thumb": 0.54,
            "Right Hand - Pinky": 0.7
        },
        "n_thresholds": {"Thumb": 0.63, "Middle": 0.43, "Pinky": 0.69},
        "r_keybinds": {
            "Right Hand - Index": None, "Right Hand - Middle": None, "Right Hand - Ring": None,
            "Right Hand - Pinky": "mouse:right", "Right Hand - Thumb": "mouse:left",
            "Left Hand - Index": "d", "Left Hand - Middle": "w", "Left Hand - Ring": "a",
            "Left Hand - Pinky": "s", "Left Hand - Thumb": "space"
        },
        "n_keybinds": {"Pinky": "mouse:left", "Thumb": "mouse:right", "Middle": "mouse:middle"}
    }


def load_config():
    defaults = get_default_config()
    file_exists = os.path.exists(CONFIG_FILE)
    updated_needed = False
    
    if file_exists:
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_data = json.load(f)
            
            for key, value in defaults.items():
                if key not in loaded_data:
                    loaded_data[key] = value
                    updated_needed = True
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if sub_key not in loaded_data[key]:
                            loaded_data[key][sub_key] = sub_val
                            updated_needed = True
            
            if updated_needed:
                print(f"Syncing missing default values to {CONFIG_FILE}...")
                save_config(loaded_data)
            
            return loaded_data
        
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading {CONFIG_FILE}: {e}. Resetting to defaults.")
            save_config(defaults)
            return defaults
    else:
        save_config(defaults)
        return defaults


def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
    except IOError as e:
        print(f"Error saving config file: {e}")