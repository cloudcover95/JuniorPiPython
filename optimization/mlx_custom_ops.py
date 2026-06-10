# MLX Custom Ops Exploration for BitNet Ternary

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

if HAS_MLX:
    @mx.custom_function
    def ternary_matmul_custom(a, b):
        """
        Custom MLX op for ternary matmul with proper gradient support.
        """
        b_tern = mx.where(b > 0.5, 1.0, mx.where(b < -0.5, -1.0, 0.0))
        return mx.matmul(a, b_tern)

    def vjp_ternary_matmul(primals, cotangents):
        a, b = primals
        g = cotangents[0]
        b_tern = mx.where(b > 0.5, 1.0, mx.where(b < -0.5, -1.0, 0.0))
        da = mx.matmul(g, mx.transpose(b_tern))
        db = mx.matmul(mx.transpose(a), g) * (mx.abs(b) < 0.5)  # Straight-through estimator style
        return da, db

    ternary_matmul_custom.vjp = vjp_ternary_matmul

    @mx.custom_function
    def fused_ternary_scale(x, scale):
        """Fused ternary quantization + scaling op"""
        tern = mx.where(x > 0.5, 1.0, mx.where(x < -0.5, -1.0, 0.0))
        return tern * scale

    class MLXCustomOps:
        def ternary_matmul(self, a, b):
            if HAS_MLX:
                return ternary_matmul_custom(a, b)
            import numpy as np
            return np.dot(a, b)

        def fused_quantize(self, x, scale=1.0):
            if HAS_MLX:
                return fused_ternary_scale(x, scale)
            import numpy as np
            tern = np.where(x > 0.5, 1.0, np.where(x < -0.5, -1.0, 0.0))
            return tern * scale

else:
    class MLXCustomOps:
        def ternary_matmul(self, a, b):
            import numpy as np
            return np.dot(a, b)

        def fused_quantize(self, x, scale=1.0):
            import numpy as np
            tern = np.where(x > 0.5, 1.0, np.where(x < -0.5, -1.0, 0.0))
            return tern * scale

print("[MLX Custom Ops] Loaded ternary + fused ops exploration")