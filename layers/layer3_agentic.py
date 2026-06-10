# Layer 3 v4 - Solid Agentic Planning

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def plan_and_execute(self, goal):
        thoughts = []
        current_goal = goal
        for step in range(5):
            thought = self.layer2.generate_with_context(f"Step {step}: {current_goal}")
            thoughts.append(thought)
            current_goal = str(thought)[:100]
        return thoughts