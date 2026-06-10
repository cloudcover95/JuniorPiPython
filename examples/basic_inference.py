# Example: Basic BitNet Inference

from bitnet.inference.engine import BitNetEngine

config = {"layer_sizes": [64, 128, 64]}
engine = BitNetEngine(config)

prompt = np.random.randn(1, 64).astype(np.float32)
output = engine.generate(prompt)
print("Output shape:", output.shape)
print("Sample output:", output[0, :5])