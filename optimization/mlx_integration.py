# MLX Integration for Apple Silicon

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class MLXBitNet:
    def to_mlx(self, weights):
        if not HAS_MLX:
            return weights
        return [mx.array(w.astype(np.float32)) for w in weights]

    def get_thermal_config(self, temp):
        if temp > 78:
            return {"max_tokens": 32, "temperature": 0.6}
        return {"max_tokens": 256, "temperature": 0.85}