# JuniorHome Swift <-> Python Bridge Example

# Python side (to be called from Swift via PyObjC or subprocess)
from bitnet.inference.engine import BitNetEngine

def swift_generate(prompt: str, max_tokens: int = 64):
    engine = BitNetEngine({"layer_sizes": [128, 256, 128]})
    return engine.generate(prompt, max_tokens=max_tokens)

# Swift side would call this via bridge