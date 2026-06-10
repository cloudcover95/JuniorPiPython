# Layer 3 v6 - Agentic Planning

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def plan(self, goal):
        thoughts = []
        for i in range(8):
            t = self.layer2.generate_with_context(f"Step {i}: {goal}")
            thoughts.append(t)
        return thoughts