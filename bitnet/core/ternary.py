# BitNet Ternary Core Operations

import numpy as np

def ternarize(weights, threshold=0.5):
    """Ternary quantization: -1, 0, +1 based on threshold"""
    abs_weights = np.abs(weights)
    ternary = np.zeros_like(weights, dtype=np.int8)
    ternary[weights > threshold] = 1
    ternary[weights < -threshold] = -1
    return ternary

def ternary_matmul(input_tensor, weight_tensor):
    """Simulated efficient ternary matmul (real version uses bit packing)"""
    # For demo: use standard matmul but with ternary weights
    ternary_weights = ternarize(weight_tensor)
    return np.dot(input_tensor, ternary_weights)

def scale_activations(activations):
    """Simple activation scaling for stability"""
    return np.clip(activations, -1.0, 1.0)