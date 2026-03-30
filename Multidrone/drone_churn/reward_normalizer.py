"""Running reward normalizer for Multidrone."""
import numpy as np

class RewardNormalizer:
    def __init__(self, clip_range=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward):
        if self.count < 2:
            return reward
        std = np.sqrt(self.var / self.count)
        normalized = (reward - self.mean) / (std + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)
