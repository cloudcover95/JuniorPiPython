# JuniorSOL + BitNet Python Integration

def get_training_plan(goal):
    from layers.layer2_contextual_brain import ContextualBrainLayer
    brain = ContextualBrainLayer(engine)
    return brain.generate_with_context(goal)