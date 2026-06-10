# Advanced Generation Example with BitNet

from bitnet.inference.engine import BitNetEngine

import numpy as np

config = {"layer_sizes": [128, 256, 128, 64]}
engine = BitNetEngine(config)

prompt = np.random.randn(1, 128).astype(np.float32)
output = engine.generate(prompt, max_tokens=32, temperature=0.8)
print("Generated token sequence:", output[:10])