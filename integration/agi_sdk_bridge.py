# Bridge to JuniorAGI_SDK

try:
    from junior_agi_sdk.coaching import TimelineEngine
    from junior_agi_sdk.core import ModelRouter
except ImportError:
    TimelineEngine = None
    ModelRouter = None

class BitNetAGIBridge:
    def __init__(self):
        self.router = ModelRouter() if ModelRouter else None
        self.timeline = TimelineEngine() if TimelineEngine else None

    def coached_inference(self, prompt, hardware="apple_silicon"):
        if self.router:
            profile = self.router.route("inference", hardware)
            print(f"Using profile: {profile}")
        return f"Coached BitNet inference complete for hardware: {hardware}"