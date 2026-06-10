# Enterprise Integration Examples

def swift_generate(prompt):
    from bitnet.inference.engine import BitNetEngine
    return BitNetEngine({"layer_sizes": [128, 256, 128]}).generate(prompt)

def solana_plan(goal):
    from layers.layer2_contextual_brain import ContextualBrainLayer
    return ContextualBrainLayer(engine).generate_with_context(goal)