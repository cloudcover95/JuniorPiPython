# MLX for Apple Silicon

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class MLXBitNet:
    def convert(self, weights):
        if not HAS_MLX:
            return weights
        return [mx.array(w.astype("float32")) for w in weights]

    def thermal_config(self, temp):
        if temp > 80:
            return {"max_tokens": 28, "temp": 0.55}
        return {"max_tokens": 256, "temp": 0.82}