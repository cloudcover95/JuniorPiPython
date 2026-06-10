# Layer 3 v3 - Improved Agentic Reasoning

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def reason(self, goal):
        context = self.layer2.generate_with_context(f"Think step by step about: {goal}")
        return context

    def multi_step_plan(self, goal, steps=4):
        results = []
        current = goal
        for i in range(steps):
            thought = self.reason(f"Step {i+1}: {current}")
            results.append(thought)
            current = thought
        return results