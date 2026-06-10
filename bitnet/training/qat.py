# QAT

import numpy as np

class QATrainer:
    def __init__(self, engine):
        self.engine = engine

    def loss(self, a, b):
        return float(np.mean((np.array(a) - np.array(b))**2))

    def step(self, lr=0.0005):
        loss = 0.0
        for i in range(len(self.engine.weights)):
            w = self.engine.weights[i].astype("float32")
            g = np.random.randn(*w.shape).astype("float32") * 0.01
            w = np.clip(w - lr * g, -3, 3)
            self.engine.weights[i] = np.round(w).astype("int8")
            loss += self.loss(w, self.engine.weights[i])
        return loss / len(self.engine.weights)