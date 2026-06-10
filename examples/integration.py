# Integration Examples
def swift_call(p): return BitNetEngine({"layer_sizes":[128,256,128]}).generate(p)
def sol_call(g): return ContextualBrainLayer(engine).generate_with_context(g)