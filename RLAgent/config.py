"""All tunable settings for the network, the Q-learning, and the game."""
from dataclasses import dataclass, field, asdict
import json
import os

# The settings the dashboard is using (written on Apply and on Load). Both modes start from them.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'settings.json')

# Settings that decide the shape of the network. Saved weights only fit the network
# they were trained with, so these always come from the checkpoint when one is loaded.
BRAIN_SETTINGS = ['state_type', 'vision_directions', 'apple_self_vision', 'hidden_layers', 'hidden_activation']


@dataclass
class Config:
    # ---- Game -------------------------------------------------------------
    grid_x: int = 20                         # keep it even: the game starts the snake at grid/2
    grid_y: int = 20

    # ---- State: what the agent sees (see environment.py) --------------------
    state_type: str = 'vision'               # vision: the same inputs as SnakeGenetics | simple: 14 yes/no facts
    vision_directions: int = 8               # vision: 8 or 4 rays
    apple_self_vision: str = 'binary'        # vision: binary | distance

    # ---- Neural network (the Q-function) -----------------------------------
    hidden_layers: list = field(default_factory=lambda: [32, 16])  # neurons per hidden layer
    hidden_activation: str = 'relu'          # relu | tanh | sigmoid
    learning_rate: float = 0.001             # Adam optimizer step size
    loss: str = 'mse'                        # mse | huber

    # ---- Q-learning --------------------------------------------------------
    gamma: float = 0.9                       # discount: how much future rewards count (0..1)
    epsilon_start: float = 1.0               # chance of a random move at the start
    epsilon_min: float = 0.01                # it never explores less than this
    epsilon_decay: float = 0.97              # epsilon is multiplied by this after every episode

    # ---- Experience replay -------------------------------------------------
    memory_size: int = 100_000               # how many past moves are remembered
    batch_size: int = 64                     # memories used per training step
    train_every: int = 1                     # train once every N moves
    target_update: int = 500                 # copy the network to the target network every N moves (0 = no target network)

    # ---- Rewards -----------------------------------------------------------
    reward_apple: float = 10.0
    reward_death: float = -10.0              # hitting the wall or itself, or starving
    reward_step: float = 0.0                 # every other move

    seed: int = 0                            # 0 = random

    # ---------------------------------------------------------------------
    @property
    def num_inputs(self):
        if self.state_type == 'simple':
            return 14                        # wall F/R/L + moving R/D/L/U + body F/R/L + apple L/R/U/D
        return self.vision_directions * 3 + 4 + 4   # rays x (wall, apple, self) + head dir + tail dir

    @property
    def num_outputs(self):
        return 3                             # one Q-value per move: [forward, right, left]

    @property
    def layer_sizes(self):
        return [self.num_inputs] + list(self.hidden_layers) + [self.num_outputs]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_last(cls):
        """The settings last applied in the dashboard, or the defaults if there are none yet."""
        if os.path.exists(SETTINGS_FILE):
            return cls.load(SETTINGS_FILE)
        return cls()

    def same_brain_as(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in BRAIN_SETTINGS)

    def with_brain_of(self, other):
        """A copy of these settings, but with the brain settings taken from `other`."""
        d = self.to_dict()
        for name in BRAIN_SETTINGS:
            d[name] = getattr(other, name)
        return Config.from_dict(d)
