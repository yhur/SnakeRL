"""Save / load training to one .npz file: settings, network weights, epsilon and history.

The replay memory is not saved (it can hold 100,000 moves); it refills quickly.
"""
import json
import numpy as np

from .config import Config
from .dqn_agent import DQNAgent
from .trainer import Trainer


def save(path, trainer):
    agent = trainer.agent
    progress = dict(episode=trainer.episode, history=trainer.history, best_score=trainer.best_score,
                    best_episode=trainer.best_episode, epsilon=agent.epsilon, moves=agent.moves,
                    train_steps=agent.train_steps, target_updates=agent.target_updates)
    weights = {f'weight_{i}': w for i, w in enumerate(agent.model.get_weights())}
    np.savez(path, config=json.dumps(trainer.cfg.to_dict()), progress=json.dumps(progress), **weights)


def load(path, settings=None):
    """A Trainer that continues from the file. The network shape always comes from the file;
    every other setting comes from `settings` when given."""
    data = np.load(path)
    saved = Config.from_dict(json.loads(str(data['config'])))
    cfg = settings.with_brain_of(saved) if settings else saved

    agent = DQNAgent(cfg)
    count = len([key for key in data.files if key.startswith('weight_')])
    agent.model.set_weights([data[f'weight_{i}'] for i in range(count)])
    agent.target_model.set_weights(agent.model.get_weights())

    progress = json.loads(str(data['progress']))
    agent.epsilon = progress['epsilon']
    agent.moves = progress['moves']
    agent.train_steps = progress['train_steps']
    agent.target_updates = progress['target_updates']

    trainer = Trainer(cfg, agent)
    trainer.episode = progress['episode']
    trainer.history = progress['history']
    trainer.best_score = progress['best_score']
    trainer.best_episode = progress['best_episode']
    return trainer
