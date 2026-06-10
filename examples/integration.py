# Integration

def swift(p): return BitNetEngine({"layer_sizes":[128,256,128]}).generate(p)
def sol(goal): return ContextualBrainLayer(engine).generate_with_context(goal)