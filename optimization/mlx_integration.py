# MLX

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class MLXBitNet:
    def convert(self, weights):
        if not HAS_MLX: return weights
        return [mx.array(w.astype("float32")) for w in weights]

    def config(self, temp):
        if temp > 80: return {"max_tokens": 32, "temperature": 0.5}
        return {"max_tokens": 2048, "temperature": 0.82}