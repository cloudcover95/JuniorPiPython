# Swift <-> Python Bridge (JuniorHome)

def generate_from_swift(prompt):
    from bitnet.inference.engine import BitNetEngine
    return BitNetEngine({"layer_sizes": [128, 256, 128]}).generate(prompt)