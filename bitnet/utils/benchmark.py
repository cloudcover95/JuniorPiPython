# BitNet Benchmarking Utilities

import time
import numpy as np
from bitnet.inference.engine import BitNetEngine

def benchmark_inference(engine, input_shape=(1, 128), runs=10):
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    times = []
    for _ in range(runs):
        start = time.time()
        _ = engine.forward(dummy_input)
        times.append(time.time() - start)
    avg_time = np.mean(times)
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    return avg_time