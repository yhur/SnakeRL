#!/usr/bin/env python
"""Teach a neural network to play SnakeGame with Deep Q-Learning (TensorFlow).

    python main.py                         dashboard: watch it learn, move by move
    python main.py --headless              no window: train as fast as possible
    python main.py --headless -e 300       ... stop after 300 episodes
    python main.py --headless --games 1    ... one game at a time, exactly like the dashboard
    python main.py --load saves/x.npz      continue from a checkpoint (either mode)

Both modes use settings.json: the settings last applied in the dashboard. With --load, the
network shape (state, hidden layers, activation) comes from the checkpoint instead.

Dashboard keys:  Space = pause / one move    R = resume    Q = quit
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')       # hide TensorFlow's start-up messages

from RLAgent.config import Config                        # noqa: E402
from RLAgent.trainer import Trainer                      # noqa: E402
from RLAgent import checkpoint                           # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--headless', action='store_true', help='train without a window')
    parser.add_argument('-e', '--episodes', type=int, default=0, help='headless: episodes to run (0 = until Ctrl+C)')
    parser.add_argument('--games', type=int, default=8, help='headless: games played at once (default: 8)')
    parser.add_argument('--save', help='headless: checkpoint file (default: saves/checkpoint_<vision|simple>.npz)')
    parser.add_argument('--config', help='JSON settings file (default: settings.json, written by the dashboard)')
    parser.add_argument('--load', help='checkpoint to continue from')
    args = parser.parse_args()

    settings = Config.load(args.config) if args.config else Config.load_last()
    if args.load:
        trainer = checkpoint.load(args.load, settings)   # network from the file, the rest from settings
    else:
        trainer = Trainer(settings)

    if args.headless:
        from RLAgent import headless
        if args.games > 1:
            from RLAgent.parallel_trainer import ParallelTrainer
            trainer = ParallelTrainer.from_trainer(trainer, args.games)
        save = args.save or f'saves/checkpoint_{trainer.cfg.state_type}.npz'
        headless.train(trainer, args.episodes, save)
    else:
        from PyQt5.QtWidgets import QApplication
        from Dashboard.main_window import Dashboard
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = Dashboard(trainer)
        window.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
