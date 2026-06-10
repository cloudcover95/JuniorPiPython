# Layer 3 v2 - Better Agentic Planning

class Layer3Agentic:
    def __init__(self, layer2):
        self.layer2 = layer2

    def plan(self, goal):
        steps = self.layer2.generate_with_context(f"Break down goal: {goal}")
        return steps

    def execute_plan(self, goal):
        plan = self.plan(goal)
        results = []
        for step in ["analyze", "gather", "act", "verify"]:
            result = self.layer2.generate_with_context(f"{step}: {goal}")
            results.append(result)
        return {"goal": goal, "plan": plan, "results": results}