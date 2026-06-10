# MLX Apple Silicon

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

    def config_for_temp(self, temp):
        if temp > 82:
            return {"max_tokens": 24, "temperature": 0.5}
        return {"max_tokens": 256, "temperature": 0.8}