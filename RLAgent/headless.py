"""Train without a window, as fast as possible. Ctrl+C stops and saves.

Works with Trainer (one game) and ParallelTrainer (several games at once).
"""
import os
import time

from . import checkpoint

SAVE_EVERY = 50     # episodes


def train(trainer, episodes, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    last_episode = trainer.episode + episodes - 1 if episodes else None
    games = len(getattr(trainer, 'envs', [trainer.env]))
    print(f'Training. Network {trainer.cfg.layer_sizes}, {games} game(s) at once. '
          f'Ctrl+C to stop and save to {save_path}')
    started, moves_at_start = time.time(), trainer.agent.moves
    printed = len(trainer.history)
    try:
        while last_episode is None or trainer.episode <= last_episode:
            trainer.step()
            for h in trainer.history[printed:]:                 # the games that just ended
                loss = f'{h["loss"]:.3f}' if h['loss'] is not None else '-'
                speed = (trainer.agent.moves - moves_at_start) / (time.time() - started)
                print(f'episode {h["episode"]:5d}   score {h["score"]:3d}   '
                      f'best {trainer.best_score:3d}   avg(50) {h["avg_score"]:6.2f}   '
                      f'epsilon {h["epsilon"]:.3f}   loss {loss}   '
                      f'{speed:5.0f} moves/s   ({h["status"]})')
                if h['episode'] % SAVE_EVERY == 0:
                    checkpoint.save(save_path, trainer)
            printed = len(trainer.history)
    except KeyboardInterrupt:
        print()
    checkpoint.save(save_path, trainer)
    print(f'Saved {save_path}')
