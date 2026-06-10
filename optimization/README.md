# Optimized MLX Metal Kernels

Performance-optimized Metal kernels for BitNet ternary operations:

- Vectorized `float4` loads
- On-the-fly ternary quantization inside kernel
- Better thread indexing and memory access patterns
- Designed for high throughput on Apple Silicon GPUs