# Hardware Detector for BitNet

import platform

def get_hardware_profile():
    system = platform.system()
    if "Darwin" in system:
        return "apple_silicon"
    elif "Linux" in system:
        return "jetson_or_pi"
    return "unknown"

class HardwareGovernor:
    def get_thermal_state(self):
        return "nominal"  # Placeholder