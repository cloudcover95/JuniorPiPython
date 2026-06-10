# Layer 3 v5 - Structured Agentic Reasoning

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def think(self, goal):
        return self.layer2.generate_with_context(f"Reason about: {goal}")

    def plan(self, goal):
        steps = []
        for i in range(6):
            step = self.think(f"Step {i+1} toward {goal}")
            steps.append(step)
        return steps