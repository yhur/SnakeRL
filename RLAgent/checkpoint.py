"""Save / load training to one .npz file: settings, network weights, epsilon and history.

The replay memory is not saved (it can hold 100,000 moves); it refills quickly.
The history (one entry per episode) is saved as one compact array per field, about 20 bytes
per episode, instead of as text. Older files with the history as text still load.
"""
import json
import numpy as np

from .config import Config
from .dqn_agent import DQNAgent
from .trainer import Trainer

# history field -> array type in the file ('status' is stored as a number, see below)
HISTORY_FIELDS = {'episode': np.int32, 'score': np.int16, 'steps': np.int32, 'avg_score': np.float32,
                  'epsilon': np.float32, 'total_reward': np.float32, 'loss': np.float32}


def history_to_arrays(history):
    arrays = {f'history_{key}': np.array([np.nan if h[key] is None else h[key] for h in history], dtype=kind)
              for key, kind in HISTORY_FIELDS.items()}
    statuses = sorted({h['status'] for h in history})                 # e.g. ['body', 'starved', 'wall']
    code = {name: i for i, name in enumerate(statuses)}
    arrays['history_status'] = np.array([code[h['status']] for h in history], dtype=np.uint8)
    return arrays, statuses


def arrays_to_history(data, statuses):
    columns = {key: data[f'history_{key}'].tolist() for key in HISTORY_FIELDS}
    columns['loss'] = [None if loss != loss else loss for loss in columns['loss']]    # NaN = no loss
    columns['status'] = [statuses[i] for i in data['history_status'].tolist()]
    keys = list(columns)
    return [dict(zip(keys, values)) for values in zip(*columns.values())]


def save(path, trainer):
    agent = trainer.agent
    history, statuses = history_to_arrays(trainer.history)
    progress = dict(episode=trainer.episode, history_statuses=statuses, best_score=trainer.best_score,
                    best_episode=trainer.best_episode, epsilon=agent.epsilon, moves=agent.moves,
                    train_steps=agent.train_steps, target_updates=agent.target_updates)
    weights = {f'weight_{i}': w for i, w in enumerate(agent.model.get_weights())}
    np.savez(path, config=json.dumps(trainer.cfg.to_dict()), progress=json.dumps(progress), **weights, **history)


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
    if 'history' in progress:                       # older files: the history as text
        trainer.history = progress['history']
    else:
        trainer.history = arrays_to_history(data, progress['history_statuses'])
    trainer.best_score = progress['best_score']
    trainer.best_episode = progress['best_episode']
    return trainer
