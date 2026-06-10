# Optimized MLX Metal Kernels with Memory Coalescing

Performance-focused Metal kernels for BitNet ternary:

- Explicit memory coalescing strategy documented in kernel
- Vectorized `float4` loads
- Coalesced reads from B matrix and writes to C matrix
- Designed for maximum memory bandwidth on Apple Silicon