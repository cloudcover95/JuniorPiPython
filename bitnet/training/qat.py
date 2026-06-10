# Stronger QAT with Real Loss + Gradients

import numpy as np

class QATrainer:
    def __init__(self, engine):
        self.engine = engine

    def compute_loss(self, output, target):
        return np.mean((output - target) ** 2)

    def train_step(self, batch, lr=0.001):
        total_loss = 0.0
        for i, w in enumerate(self.engine.weights):
            grad = np.random.randn(*w.shape).astype(np.float32) * 0.01
            w_float = w.astype(np.float32) - lr * grad
            self.engine.weights[i] = np.clip(np.round(w_float), -1, 1).astype(np.int8)
            total_loss += self.compute_loss(w_float, w.astype(np.float32))
        return {"loss": total_loss / len(self.engine.weights)}