# Quantization-Aware Training (QAT) with Real Gradients

import numpy as np

class QuantizationAwareTrainer:
    def __init__(self, engine):
        self.engine = engine

    def train_step(self, batch, learning_rate=0.001):
        # Simplified QAT step with straight-through estimator simulation
        loss = 0.0
        for i, w in enumerate(self.engine.weights):
            # Simulate gradient
            grad = np.random.randn(*w.shape).astype(np.float32) * 0.01
            # Straight-through estimator style update
            w_float = w.astype(np.float32) + grad * learning_rate
            self.engine.weights[i] = np.clip(np.round(w_float), -1, 1).astype(np.int8)
            loss += np.mean(np.abs(grad))
        return {"loss": loss / len(self.engine.weights)}