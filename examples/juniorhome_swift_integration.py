# Concrete Swift <-> Python Integration Example (JuniorHome)

"""
# In Swift (JuniorHome):
# BitNetPythonBridge.call("generate", prompt: prompt)

# In Python:
from bitnet.inference.engine import BitNetEngine

def handle_swift_request(prompt):
    engine = BitNetEngine({"layer_sizes": [128, 256, 128]})
    return engine.generate(prompt)
"""