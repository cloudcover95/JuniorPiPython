# Layer 2 Bridge to JuniorAGI_SDK

try:
    from junior_agi_sdk.coaching import TimelineEngine
except ImportError:
    TimelineEngine = None

class Layer2AGIBridge:
    def __init__(self, contextual_brain):
        self.brain = contextual_brain
        self.timeline = TimelineEngine() if TimelineEngine else None

    def run_coached_brain_session(self, prompt, hardware):
        self.brain.set_hardware(hardware)
        if self.timeline:
            self.timeline.create_timeline("layer2_session", ["context_load", "inference", "context_update"])
        return self.brain.generate_with_context(prompt)