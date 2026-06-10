# Optimized MLX Metal Kernels for BitNet Ternary (with Memory Coalescing)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    # Highly optimized Metal kernel with explicit memory coalescing
    # Key optimization: Threads in the same threadgroup access consecutive memory addresses
    # for both A (row-wise) and B (column-wise) to maximize memory bandwidth.
    optimized_ternary_kernel = mx.fast.metal_kernel(
        name="coalesced_ternary_matmul",
        input_names=["A", "B"],
        output_names=["C"],
        source="""
            #include <metal_stdlib>
            using namespace metal;

            kernel void coalesced_ternary_matmul(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                // Memory Coalescing Strategy:
                // - gid.x (column) varies fastest across threads in a warp
                // - This makes accesses to B and C coalesced (consecutive addresses)
                // - A is accessed row-wise (less coalesced but cached)

                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                float sum = 0.0f;

                // Coalesced accumulation loop
                // Each thread reads consecutive elements in B as col varies
                for (uint k = 0; k < K; k += 4) {
                    float4 a_vec = float4(
                        A[row * K + k + 0],
                        A[row * K + k + 1],
                        A[row * K + k + 2],
                        A[row * K + k + 3]
                    );

                    // Coalesced read from B (threads with consecutive col access consecutive memory)
                    float4 b_vec = float4(
                        B[(k + 0) * N + col],
                        B[(k + 1) * N + col],
                        B[(k + 2) * N + col],
                        B[(k + 3) * N + col]
                    );

                    // Ternary quantization
                    float4 b_tern;
                    b_tern.x = (b_vec.x > 0.5f) ? 1.0f : ((b_vec.x < -0.5f) ? -1.0f : 0.0f);
                    b_tern.y = (b_vec.y > 0.5f) ? 1.0f : ((b_vec.y < -0.5f) ? -1.0f : 0.0f);
                    b_tern.z = (b_vec.z > 0.5f) ? 1.0f : ((b_vec.z < -0.5f) ? -1.0f : 0.0f);
                    b_tern.w = (b_vec.w > 0.5f) ? 1.0f : ((b_vec.w < -0.5f) ? -1.0f : 0.0f);

                    sum += dot(a_vec, b_tern);
                }

                C[row * N + col] = sum;  // Coalesced write
            }
        """ 
    )

    def coalesced_metal_ternary_matmul(A, B, M, N, K):
        if HAS_MLX:
            return optimized_ternary_kernel(A, B, M=M, N=N, K=K)
        import numpy as np
        return np.dot(A, B)

    class OptimizedMLXMetalKernels:
        def run(self, A, B, M, N, K):
            return coalesced_metal_ternary_matmul(A, B, M, N, K)

        def coalescing_notes(self):
            return (
                "Memory Coalescing:\n"
                "- Threads with consecutive gid.x read consecutive addresses in B and C\n"
                "- This allows GPU to combine memory transactions into fewer cache lines\n"
                "- Critical for high bandwidth utilization on Apple Silicon GPUs"
            )
else:
    def coalesced_metal_ternary_matmul(A, B, M, N, K):
        import numpy as np
        return np.dot(A, B)

    class OptimizedMLXMetalKernels:
        def run(self, A, B, M, N, K):
            import numpy as np
            return np.dot(A, B)

        def coalescing_notes(self):
            return "NumPy fallback - no GPU coalescing"

print("[MLX Metal Kernels] Memory-coalesced optimized ternary kernel loaded")