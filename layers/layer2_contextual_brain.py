# Layer 2: Contextual Brain for BitNet Ternary (Mobile Silicon Optimized)

import numpy as np

class ContextualBrainLayer:
    """
    Layer 2 - Contextual Brain
    Sits on top of raw BitNet inference (Layer 1).
    Provides memory, context, state tracking, and intelligent routing.
    Optimized for mobile silicon (Apple Silicon, ARM edge).
    """

    def __init__(self, bitnet_engine):
        self.engine = bitnet_engine
        self.context_memory = []          # Simple contextual memory
        self.state = {"session_id": None, "hardware": "unknown"}

    def set_hardware(self, hardware_profile):
        self.state["hardware"] = hardware_profile

    def add_context(self, context_item):
        self.context_memory.append(context_item)
        # Keep memory bounded for mobile efficiency
        if len(self.context_memory) > 50:
            self.context_memory.pop(0)

    def generate_with_context(self, prompt, max_tokens=64):
        # Inject context into prompt (simplified)
        contextual_prompt = prompt
        if self.context_memory:
            context_str = " | ".join([str(c) for c in self.context_memory[-5:]])
            contextual_prompt = f"{context_str} || {prompt}"

        # Call into Layer 1 (BitNetEngine)
        output = self.engine.generate(contextual_prompt, max_tokens=max_tokens)
        self.add_context(output)
        return output

    def get_brain_state(self):
        return {
            "context_length": len(self.context_memory),
            "hardware": self.state["hardware"],
            "session_active": self.state["session_id"] is not None
        }