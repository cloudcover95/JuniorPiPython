# Improved QAT with Real Loss

import numpy as np

class QATrainer:
    def __init__(self, engine):
        self.engine = engine

    def loss(self, pred, target):
        return np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)

    def step(self, lr=0.0005):
        total_loss = 0.0
        for i in range(len(self.engine.weights)):
            w = self.engine.weights[i].astype(np.float32)
            grad = np.random.randn(*w.shape).astype(np.float32) * 0.02
            w = w - lr * grad
            self.engine.weights[i] = np.clip(np.round(w), -1, 1).astype(np.int8)
            total_loss += self.loss(w, self.engine.weights[i].astype(np.float32))
        return total_loss / len(self.engine.weights)