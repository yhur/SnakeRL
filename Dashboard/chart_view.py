"""Score per episode, and its average over the last 50 episodes. Red dashed lines: rollbacks.

A long run has hundreds of thousands of episodes, far more than the chart has pixels. So the values
are kept as arrays and reduced to one point per pixel column: the highest score and the mean
average score of the episodes in that column.
"""
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPolygonF

SERIES = [('score', QColor(0, 220, 90), 'score'),
          ('avg_score', QColor(80, 170, 255), 'avg score')]


class ChartView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setMinimumHeight(140)
        self._reset()

    def _reset(self):
        self.cached_of = None                  # the history list the arrays below were made from
        self.values = {key: np.zeros(0) for key, _, _ in SERIES}
        self.rollbacks = []                    # episode indexes

    def set_history(self, history):
        self.history = history
        self.update()

    def _sync(self):
        """Add the episodes that were appended since the last paint (or start over for a new history)."""
        hist = self.history
        if hist is not self.cached_of or len(hist) < len(self.values['score']):
            self._reset()
            self.cached_of = hist
        start = len(self.values['score'])
        new = hist[start:]
        if new:
            for key, _, _ in SERIES:
                self.values[key] = np.concatenate([self.values[key], [hh[key] for hh in new]])
            self.rollbacks += [start + i for i, hh in enumerate(new) if hh.get('rollback')]

    @staticmethod
    def _per_column(values, columns, reduce):
        """At most one value per pixel column."""
        if len(values) <= columns:
            return values
        edges = np.linspace(0, len(values), columns + 1).astype(int)[:-1]
        return reduce(values, edges)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 26, 30))
        f = QFont('Helvetica')
        f.setPixelSize(11)
        p.setFont(f)
        l, r, t, b = 36, 10, 20, 20
        w, h = self.width() - l - r, self.height() - t - b
        p.setPen(QColor(70, 70, 70))
        p.drawRect(QRectF(l, t, w, h))
        x = l
        for key, col, name in SERIES:
            p.setPen(col)
            p.drawText(QRectF(x, 2, 120, 16), Qt.AlignLeft, '— ' + name)
            x += 100
        hist = self.history
        if len(hist) < 1:
            return
        self._sync()
        n = len(self.values['score'])
        ymax = max(1, int(self.values['score'].max()))
        p.setPen(QColor(150, 150, 150))
        p.drawText(QRectF(0, t - 6, l - 4, 12), Qt.AlignRight, str(ymax))
        p.drawText(QRectF(0, t + h - 6, l - 4, 12), Qt.AlignRight, '0')
        p.drawText(QRectF(l, t + h + 2, w, 16), Qt.AlignRight, f'episode {hist[-1]["episode"]}')
        p.setPen(QPen(QColor(255, 90, 90), 1, Qt.DashLine))      # rollbacks (Trainer.check_block)
        for i in self.rollbacks:
            x = l + w * (i / max(1, n - 1))
            p.drawLine(QPointF(x, t), QPointF(x, t + h))
        columns = max(1, int(w))
        for key, col, _ in SERIES:
            reduce = np.maximum.reduceat if key == 'score' else \
                (lambda v, e: np.add.reduceat(v, e) / np.diff(np.append(e, len(v))))
            ys = self._per_column(self.values[key], columns, reduce)
            m = len(ys)
            poly = QPolygonF([QPointF(l + w * (i / max(1, m - 1)), t + h - h * y / ymax)
                              for i, y in enumerate(ys)])
            p.setPen(QPen(col, 1.6))
            p.drawPolyline(poly)
