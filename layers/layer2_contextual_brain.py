# Layer 2 v4 - Stronger Attention + Longer Context

import numpy as np

class ContextualBrainLayer:
    def __init__(self, engine, max_context=512):
        self.engine = engine
        self.memory = []
        self.max_context = max_context

    def add_context(self, item):
        self.memory.append(item)
        if len(self.memory) > self.max_context:
            self.memory = self.memory[-self.max_context:]

    def _attention(self, query):
        if not self.memory:
            return query
        recent = self.memory[-30:]
        scores = []
        q = np.array(query).flatten()
        for item in recent:
            m = np.array(item).flatten()[:len(q)]
            scores.append(np.dot(q, m))
        scores = np.array(scores)
        weights = np.exp(scores - np.max(scores) + 1e-8)
        weights /= np.sum(weights)
        context_vec = np.zeros_like(q)
        for w, item in zip(weights, recent):
            context_vec += w * np.array(item).flatten()[:len(q)]
        return q + 0.35 * context_vec

    def generate_with_context(self, prompt, max_tokens=256):
        enhanced = self._attention(prompt) if isinstance(prompt, np.ndarray) else prompt
        return self.engine.generate(enhanced, max_tokens=max_tokens)