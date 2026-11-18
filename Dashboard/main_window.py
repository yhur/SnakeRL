"""The dashboard: watch Deep Q-Learning train, move by move.

The training loop is in RLAgent/trainer.py. This window calls it one move at a time:

    for each episode (one game):
        until the game ends:                    <- one_move(): trainer.step()
            choose a move (sometimes a random one), make it,
            remember it, train on a random batch of memories
        explore a little less                   <- game_finished()

Keys:  Space = pause, or one move while paused    R = resume    Q = quit
"""
import os
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel,
                             QPushButton, QComboBox, QSplitter, QFileDialog, QMessageBox, QApplication,
                             QCheckBox, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QEvent

from RLAgent.trainer import Trainer, AVERAGE_OVER
from RLAgent.config import SETTINGS_FILE
from RLAgent.environment import state_labels, ACTION_NAMES as MOVE_NAMES
from RLAgent import checkpoint
from .nn_view import NetworkView
from .board_view import BoardView
from .chart_view import ChartView
from .config_panel import ConfigPanel, ComboBox

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saves')
SPEEDS = [('1 move/s', 1), ('5 moves/s', 5), ('15 moves/s', 15), ('30 moves/s', 30), ('60 moves/s', 60),
          ('200 moves/s', 200), ('1000 moves/s', 1000), ('Max', 0)]


class Dashboard(QMainWindow):
    def __init__(self, trainer):
        super().__init__()
        self.setWindowTitle('SnakeRL — Deep Q-Learning')
        self.trainer = trainer    # the agent, its game, and the training loop
        self.paused = False
        self.move_credit = 0.0    # moves owed to the speed setting (see on_timer)
        self.last_time = time.time()

        self.build_window()

        QApplication.instance().installEventFilter(self)   # catch Space / R / Q everywhere
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)                               # about 60 times a second
        self.start_game()

    # =====================================================================
    #   The training loop
    # =====================================================================
    def start_game(self):
        """Show the game the trainer is playing now."""
        self.board.show_game(self.trainer.env)

    def one_move(self):
        game_over = self.trainer.step()      # choose a move, make it, remember it, learn
        if game_over:
            self.game_finished()

    def game_finished(self):
        """The trainer has already recorded the game and started a new one."""
        self.chart.set_history(self.trainer.history)
        self.start_game()

    def on_timer(self):
        """Called about 60 times a second. Makes as many moves as the chosen speed asks for."""
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        if self.paused:
            return
        speed = self.speed.currentData()
        if speed == 0:
            # Max: keep moving for 12 milliseconds, then let the screen redraw
            while time.time() - now < 0.012:
                self.one_move()
        else:
            self.move_credit += speed * elapsed
            while self.move_credit >= 1:
                self.move_credit -= 1
                self.one_move()
        self.refresh()

    # =====================================================================
    #   Keys
    # =====================================================================
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and self.isActiveWindow():
            if event.key() == Qt.Key_Space:
                self.pause_or_step()
                return True
            if event.key() == Qt.Key_R:
                self.resume()
                return True
            if event.key() == Qt.Key_Q:
                self.close()
                return True
        return super().eventFilter(obj, event)

    def pause_or_step(self):
        if self.paused:
            self.one_move()          # already paused: make one move, and learn from it
        self.paused = True
        self.refresh()

    def resume(self):
        self.paused = False
        self.move_credit = 0
        self.refresh()

    # =====================================================================
    #   Screen
    # =====================================================================
    def refresh(self):
        trainer, agent, game, s = self.trainer, self.trainer.agent, self.trainer.env, self.stat
        s['episode'].setText(str(trainer.episode))
        s['moves'].setText(f'{agent.moves:,}')
        s['epsilon'].setText(f'{agent.epsilon:.3f}')
        s['memory'].setText(f'{len(agent.memory):,} / {agent.memory.capacity:,}')
        s['train_steps'].setText(f'{agent.train_steps:,}')
        s['target'].setText(f'{agent.target_updates:,}' if trainer.cfg.target_update else 'off')
        if trainer.history:
            last = trainer.history[-1]
            s['best'].setText(f'{trainer.best_score}  (ep. {trainer.best_episode})')
            s['last_avg'].setText(f'{last["avg_score"]:.2f}')
            s['loss'].setText(f'{last["loss"]:.4f}' if last['loss'] is not None else '-')

        q_values = agent.q_values(trainer.state)
        self.board.q_values = q_values if self.paused else None    # stepping: show them on the board
        s['score'].setText(str(game.score))
        s['steps'].setText(str(game.steps))
        s['return'].setText(f'{trainer.episode_return:+.2f}')
        self.show_last_move()
        s['next'].setText(MOVE_NAMES[int(q_values.argmax())])
        s['outputs'].setText(' / '.join(f'{v:.2f}' for v in q_values))
        s['previous'].setText(trainer.previous)
        self.nn_view.show_brain(agent.brain(trainer.state), state_labels(trainer.cfg), MOVE_NAMES)

        if self.paused:
            self.state.setText('PAUSED — Space: one move, R: resume')
            self.state.setStyleSheet('font-weight: bold; color: #e6a800;')
        else:
            self.state.setText('RUNNING — Space: pause, Q: quit')
            self.state.setStyleSheet('font-weight: bold; color: #2bb24c;')
        self.board.update()
        self.nn_view.update()

    def show_last_move(self):
        """The move just made, as the agent remembers it, and the Q-learning target it gives."""
        move, s = self.trainer.last_move, self.stat
        if move is None:
            return
        how = 'random' if move['explored'] else 'best Q'
        s['last_move'].setText(f'{MOVE_NAMES[move["action"]]}  ({how})')
        s['reward'].setText(f'{move["reward"]:+g}' + (f'  ({move["status"]})' if move['done'] else ''))
        r, gamma = move['reward'], self.trainer.cfg.gamma
        if move['done']:
            s['q_target'].setText(f'{r:+g}  (game over)')
        else:
            future = self.trainer.agent.best_future_q(move['next_state'])
            s['q_target'].setText(f'{r:+g} + {gamma:g}×{future:.2f} = {r + gamma * future:.2f}')

    def build_window(self):
        cfg = self.trainer.cfg
        self.nn_view = NetworkView()
        self.board = BoardView(cfg)
        self.chart = ChartView()
        self.chart.set_history(self.trainer.history)
        self.panel = ConfigPanel(cfg)
        self.panel.applied.connect(self.restart)

        # tool bar
        pause = QPushButton('Pause / Step  [Space]')
        resume = QPushButton('Resume  [R]')
        quit_ = QPushButton('Quit  [Q]')
        save = QPushButton('Save')
        load = QPushButton('Load...')
        pause.clicked.connect(self.pause_or_step)
        resume.clicked.connect(self.resume)
        quit_.clicked.connect(self.close)
        save.clicked.connect(self.save)
        load.clicked.connect(self.load_dialog)
        self.speed = ComboBox()
        for label, value in SPEEDS:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(3)
        vision = QCheckBox('Vision rays')
        vision.setChecked(True)
        vision.toggled.connect(self.toggle_vision)
        self.learning = QCheckBox('Learning')
        self.learning.setToolTip('Off: no random moves and no training, just watch what it has learned')
        self.learning.setChecked(True)
        self.learning.toggled.connect(self.toggle_learning)
        self.state = QLabel()
        bar = QHBoxLayout()
        for w in (pause, resume, quit_, self.speed, vision, self.learning, save, load):
            w.setFocusPolicy(Qt.NoFocus)
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.state)

        # statistics
        self.stat = {}
        left = [('episode', 'Episode'), ('moves', 'Moves'), ('epsilon', 'Epsilon (random)'),
                ('memory', 'Replay memory'), ('train_steps', 'Training steps'), ('target', 'Target net updates'),
                ('best', 'Best score ever'), ('last_avg', f'Avg score, last {AVERAGE_OVER}'),
                ('loss', 'Loss, last episode')]
        right = [('score', 'Score'), ('steps', 'Steps'), ('return', 'Total reward'), ('last_move', 'Last move'),
                 ('reward', 'Reward'), ('q_target', 'Q target r+γ·maxQ′'), ('outputs', 'Q-values F / R / L'),
                 ('next', 'Next move (best Q)'), ('previous', 'Previous game')]
        stats = QGroupBox('Training')
        grid = QGridLayout(stats)
        for column, rows in ((0, left), (2, right)):
            for row, (key, label) in enumerate(rows):
                grid.addWidget(QLabel(f'<b>{label}:</b>'), row, column)
                value = QLabel('-')
                value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # text changes don't resize panes
                grid.addWidget(value, row, column + 1)
                self.stat[key] = value
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        center = QWidget()
        column = QVBoxLayout(center)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.board, 5)
        column.addWidget(stats, 0)
        column.addWidget(self.chart, 2)

        panes = QSplitter(Qt.Horizontal)
        panes.addWidget(self.nn_view)
        panes.addWidget(center)
        panes.addWidget(self.panel)
        panes.setSizes([700, 560, 360])
        panes.setChildrenCollapsible(False)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(bar)
        layout.addWidget(panes, 1)
        self.setCentralWidget(root)
        self.resize(1680, 980)

    def toggle_vision(self, on):
        self.board.show_vision = on
        self.board.update()

    def toggle_learning(self, on):
        self.trainer.learning = on
        self.refresh()

    # =====================================================================
    #   Restart, save, load
    # =====================================================================
    def restart(self, cfg):
        """Apply the panel's settings. If the network shape is unchanged, the agent can be kept."""
        box = QMessageBox(self)
        box.setWindowTitle('Apply settings')
        if cfg.same_brain_as(self.trainer.cfg):
            box.setText('Keep the agent (what it has learned and its memory) and use the new settings\n'
                        'from now on, or start again with a new, untrained agent?')
            keep = box.addButton('Keep agent', QMessageBox.AcceptRole)
        else:
            box.setText('The network shape changed (state, layers, neurons or activation),\n'
                        'so what the agent learned cannot be kept. Start a new, untrained agent?')
            keep = None
        new = box.addButton('New agent', QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()

        if keep is not None and box.clickedButton() == keep:
            self.trainer.use_settings(cfg)         # same agent, new rules, a new game
            self.start_game()
        elif box.clickedButton() == new:
            self.use(Trainer(cfg))
        else:
            return
        cfg.save(SETTINGS_FILE)                    # headless mode uses these settings too
        self.statusBar().showMessage(f'Settings saved to {SETTINGS_FILE}', 6000)
        self.refresh()

    def use(self, trainer):
        trainer.learning = self.learning.isChecked()
        self.trainer = trainer
        self.chart.set_history(trainer.history)
        self.start_game()
        self.refresh()

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f'checkpoint_{self.trainer.cfg.state_type}_ep{self.trainer.episode}.npz')
        checkpoint.save(path, self.trainer)
        self.statusBar().showMessage(f'Saved {path}', 6000)

    def load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load checkpoint', SAVE_DIR, 'Checkpoint (*.npz)')
        if path:
            trainer = checkpoint.load(path, self.trainer.cfg)   # network from the file, everything else as now
            self.panel.load_config(trainer.cfg)
            trainer.cfg.save(SETTINGS_FILE)
            self.use(trainer)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
