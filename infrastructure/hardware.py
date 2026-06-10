# Hardware Abstraction Layer (No bloat)

import platform

class HardwareAbstraction:
    def get_profile(self):
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin":
            return "apple_silicon"
        elif "aarch64" in machine or "arm" in machine:
            return "arm_edge"
        else:
            return "x86"

    def get_thermal_state(self):
        # Placeholder - real version would read sensors
        return "nominal"

    def supports_ternary(self):
        return True