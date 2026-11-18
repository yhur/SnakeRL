"""Experience replay: remember past moves and learn from a random handful of them.

Learning only from the move just made is unstable: consecutive moves look almost the
same, so the network keeps over-fitting to whatever just happened. Instead, every move
(state, action, reward, next_state, done) goes into a big memory, and each training step
uses a random batch from it, old and new moves mixed.
"""
import random
from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.memories = deque(maxlen=capacity)     # when full, the oldest memory is dropped

    def __len__(self):
        return len(self.memories)

    @property
    def capacity(self):
        return self.memories.maxlen

    def add(self, state, action, reward, next_state, done):
        self.memories.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """A random batch, as five arrays: states, actions, rewards, next_states, dones."""
        batch = random.sample(self.memories, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), np.array(actions),
                np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def resize(self, capacity):
        """Change the capacity, keeping the newest memories."""
        self.memories = deque(self.memories, maxlen=capacity)
