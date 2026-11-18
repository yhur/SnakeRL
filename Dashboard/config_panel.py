"""Editable settings. 'Apply' uses them, keeping what the agent learned when the network shape is unchanged."""
import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QPushButton, QLabel, QScrollArea)
from PyQt5.QtCore import pyqtSignal, Qt

from RLAgent.config import Config


class _FocusWheel:
    """Mouse wheel changes the value only after the field is clicked; otherwise it scrolls the panel."""
    def _init_focus(self):
        self.setFocusPolicy(Qt.StrongFocus)    # the wheel alone never gives focus

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()                     # passes the wheel on to the scroll area


class SpinBox(_FocusWheel, QSpinBox):
    def __init__(self, *a):
        super().__init__(*a)
        self._init_focus()


class DoubleSpinBox(_FocusWheel, QDoubleSpinBox):
    def __init__(self, *a):
        super().__init__(*a)
        self._init_focus()


class ComboBox(_FocusWheel, QComboBox):
    def __init__(self, *a):
        super().__init__(*a)
        self._init_focus()


def _spin(lo, hi, val, step=1):
    s = SpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _dspin(lo, hi, val, step, decimals=3):
    s = DoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _combo(items, val):
    c = ComboBox()
    for label, data in items:
        c.addItem(label, data)
    c.setCurrentIndex(max(0, c.findData(val)))
    return c


class ConfigPanel(QScrollArea):
    applied = pyqtSignal(object)          # emits a Config

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(330)
        body = QWidget()
        self.setWidget(body)
        v = QVBoxLayout(body)

        # ---------------- Network ----------------
        nn = QGroupBox('Neural Network')
        fn = QFormLayout(nn)
        self.state_type = _combo([('Vision', 'vision'), ('Simple (14 facts)', 'simple')],
                                 cfg.state_type)
        self.vision = _combo([('8 directions', 8), ('4 directions', 4)], cfg.vision_directions)
        self.as_vision = _combo([('Binary', 'binary'), ('1 / distance', 'distance')], cfg.apple_self_vision)
        fn.addRow('State (inputs)', self.state_type)
        fn.addRow('Snake vision', self.vision)
        fn.addRow('Apple/Self vision', self.as_vision)
        self.n_layers = _spin(0, 6, len(cfg.hidden_layers))
        self.n_layers.valueChanged.connect(self._rebuild_layer_spins)
        fn.addRow('Hidden layers', self.n_layers)
        self.layer_box = QWidget()
        self.layer_form = QFormLayout(self.layer_box)
        self.layer_form.setContentsMargins(0, 0, 0, 0)
        self.layer_spins = []
        fn.addRow(self.layer_box)
        self.hidden_act = _combo([('ReLU', 'relu'), ('Tanh', 'tanh'), ('Sigmoid', 'sigmoid')],
                                 cfg.hidden_activation)
        self.lr = _dspin(0.00001, 0.1, cfg.learning_rate, 0.0005, 5)
        self.loss = _combo([('Mean squared error', 'mse'), ('Huber', 'huber')], cfg.loss)
        fn.addRow('Hidden activation', self.hidden_act)
        fn.addRow('Learning rate', self.lr)
        fn.addRow('Loss', self.loss)
        self.arch_label = QLabel()
        fn.addRow('Architecture', self.arch_label)
        v.addWidget(nn)

        # ---------------- Q-learning ----------------
        ql = QGroupBox('Q-Learning')
        fq = QFormLayout(ql)
        self.gamma = _dspin(0, 0.999, cfg.gamma, 0.01, 3)
        self.eps_start = _dspin(0, 1, cfg.epsilon_start, 0.05, 2)
        self.eps_min = _dspin(0, 1, cfg.epsilon_min, 0.01, 3)
        self.eps_decay = _dspin(0.5, 1, cfg.epsilon_decay, 0.005, 4)
        fq.addRow('Discount γ', self.gamma)
        fq.addRow('Epsilon start', self.eps_start)
        fq.addRow('Epsilon minimum', self.eps_min)
        fq.addRow('Epsilon decay / episode', self.eps_decay)
        self.q_note = QLabel()
        self.q_note.setWordWrap(True)
        self.q_note.setStyleSheet('color: #999;')
        fq.addRow(self.q_note)
        v.addWidget(ql)

        # ---------------- Experience replay ----------------
        er = QGroupBox('Experience Replay')
        fe = QFormLayout(er)
        self.memory = _spin(100, 1_000_000, cfg.memory_size, 1000)
        self.batch = _spin(1, 1024, cfg.batch_size, 8)
        self.train_every = _spin(1, 100, cfg.train_every)
        self.target_update = _spin(0, 100_000, cfg.target_update, 100)
        self.target_update.setSpecialValueText('off')
        fe.addRow('Memory size', self.memory)
        fe.addRow('Batch size', self.batch)
        fe.addRow('Train every N moves', self.train_every)
        fe.addRow('Target net update (moves)', self.target_update)
        v.addWidget(er)

        # ---------------- Rewards ----------------
        rw = QGroupBox('Rewards')
        fr = QFormLayout(rw)
        self.r_apple = _dspin(-100, 100, cfg.reward_apple, 1, 2)
        self.r_death = _dspin(-100, 100, cfg.reward_death, 1, 2)
        self.r_step = _dspin(-10, 10, cfg.reward_step, 0.01, 2)
        fr.addRow('Eat an apple', self.r_apple)
        fr.addRow('Die (wall, body, starve)', self.r_death)
        fr.addRow('Any other move', self.r_step)
        v.addWidget(rw)

        # ---------------- Game ----------------
        gm = QGroupBox('Game')
        fg = QFormLayout(gm)
        self.gx = _spin(6, 80, cfg.grid_x, 2)      # even only: the game starts the snake at grid/2
        self.gy = _spin(6, 80, cfg.grid_y, 2)
        self.seed = _spin(0, 2 ** 31 - 1, cfg.seed)
        self.seed.setSpecialValueText('random')
        fg.addRow('Grid width', self.gx)
        fg.addRow('Grid height', self.gy)
        fg.addRow('Random seed', self.seed)
        v.addWidget(gm)

        row = QHBoxLayout()
        self.apply_btn = QPushButton('Apply')
        self.apply_btn.setStyleSheet('font-weight: bold; padding: 6px;')
        self.apply_btn.clicked.connect(lambda: self.applied.emit(self.config()))
        reset = QPushButton('Defaults')
        reset.clicked.connect(lambda: self.load_config(Config()))
        row.addWidget(reset)
        row.addWidget(self.apply_btn)
        v.addLayout(row)
        v.addStretch(1)

        for w in self._all_inputs():
            sig = getattr(w, 'valueChanged', None) or w.currentIndexChanged
            sig.connect(self._refresh_labels)
        self._rebuild_layer_spins(len(cfg.hidden_layers), cfg.hidden_layers)

    # ------------------------------------------------------------------
    def _all_inputs(self):
        return [self.state_type, self.vision, self.as_vision, self.gamma, self.eps_start, self.eps_min,
                self.eps_decay, self.target_update]

    def _rebuild_layer_spins(self, n, values=None):
        old = [s.value() for s in self.layer_spins]
        values = list(values) if values is not None else old
        while self.layer_form.rowCount():
            self.layer_form.removeRow(0)
        self.layer_spins = []
        for i in range(n):
            val = values[i] if i < len(values) else (values[-1] if values else 16)
            s = _spin(1, 256, val)
            s.valueChanged.connect(self._refresh_labels)
            self.layer_form.addRow(f'  Neurons in layer {i + 1}', s)
            self.layer_spins.append(s)
        self._refresh_labels()

    def _refresh_labels(self, *_):
        cfg = self.config()
        sizes = cfg.layer_sizes
        n = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
        self.arch_label.setText(f'{sizes}  ({n} weights)')
        vision = cfg.state_type == 'vision'
        self.vision.setEnabled(vision)
        self.as_vision.setEnabled(vision)

        # When does epsilon reach its minimum?
        if cfg.epsilon_decay < 1 and 0 < cfg.epsilon_min < cfg.epsilon_start:
            episodes = math.ceil(math.log(cfg.epsilon_min / cfg.epsilon_start) / math.log(cfg.epsilon_decay))
            when = f'Epsilon reaches {cfg.epsilon_min} after about {episodes} episodes.'
        else:
            when = 'Epsilon stays the same.'
        horizon = f'{1 / (1 - cfg.gamma):.0f}' if cfg.gamma < 1 else 'endless'
        self.q_note.setText(f'{when} With γ = {cfg.gamma}, a reward counts about {horizon} moves ahead. '
                            'Target net "off" learns against the network itself.')

    def config(self) -> Config:
        return Config(
            grid_x=self.gx.value(), grid_y=self.gy.value(),
            state_type=self.state_type.currentData(), vision_directions=self.vision.currentData(),
            apple_self_vision=self.as_vision.currentData(),
            hidden_layers=[s.value() for s in self.layer_spins],
            hidden_activation=self.hidden_act.currentData(), learning_rate=self.lr.value(),
            loss=self.loss.currentData(), gamma=self.gamma.value(),
            epsilon_start=self.eps_start.value(), epsilon_min=self.eps_min.value(),
            epsilon_decay=self.eps_decay.value(), memory_size=self.memory.value(),
            batch_size=self.batch.value(), train_every=self.train_every.value(),
            target_update=self.target_update.value(), reward_apple=self.r_apple.value(),
            reward_death=self.r_death.value(), reward_step=self.r_step.value(), seed=self.seed.value())

    def load_config(self, cfg: Config):
        for w in self.findChildren(QWidget):
            w.blockSignals(True)
        self.n_layers.setValue(len(cfg.hidden_layers))
        for combo, val in [(self.hidden_act, cfg.hidden_activation), (self.loss, cfg.loss),
                           (self.state_type, cfg.state_type), (self.vision, cfg.vision_directions),
                           (self.as_vision, cfg.apple_self_vision)]:
            combo.setCurrentIndex(max(0, combo.findData(val)))
        for spin, val in [(self.lr, cfg.learning_rate), (self.gamma, cfg.gamma),
                          (self.eps_start, cfg.epsilon_start), (self.eps_min, cfg.epsilon_min),
                          (self.eps_decay, cfg.epsilon_decay), (self.memory, cfg.memory_size),
                          (self.batch, cfg.batch_size), (self.train_every, cfg.train_every),
                          (self.target_update, cfg.target_update), (self.r_apple, cfg.reward_apple),
                          (self.r_death, cfg.reward_death), (self.r_step, cfg.reward_step),
                          (self.gx, cfg.grid_x), (self.gy, cfg.grid_y), (self.seed, cfg.seed)]:
            spin.setValue(val)
        for w in self.findChildren(QWidget):
            w.blockSignals(False)
        self._rebuild_layer_spins(len(cfg.hidden_layers), cfg.hidden_layers)
