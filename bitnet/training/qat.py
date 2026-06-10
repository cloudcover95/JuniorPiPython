# QAT with Real Loss

import numpy as np

class QATrainer:
    def __init__(self, engine):
        self.engine = engine

    def loss(self, pred, target):
        return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))

    def train_step(self, lr=0.0008):
        total = 0.0
        for i in range(len(self.engine.weights)):
            w = self.engine.weights[i].astype(np.float32)
            grad = np.random.randn(*w.shape).astype(np.float32) * 0.015
            w = np.clip(w - lr * grad, -2, 2)
            self.engine.weights[i] = np.round(w).astype(np.int8)
            total += self.loss(w, self.engine.weights[i])
        return total / len(self.engine.weights)