# Layer 3 v7

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def plan(self, goal):
        return [self.layer2.generate_with_context(f"Step {i}: {goal}") for i in range(8)]