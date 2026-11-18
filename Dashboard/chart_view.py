"""Score per episode, and its average over the last 50 episodes."""
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

    def set_history(self, history):
        self.history = history
        self.update()

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
        ymax = max(1, max(hh['score'] for hh in hist))
        n = len(hist)
        p.setPen(QColor(150, 150, 150))
        p.drawText(QRectF(0, t - 6, l - 4, 12), Qt.AlignRight, str(ymax))
        p.drawText(QRectF(0, t + h - 6, l - 4, 12), Qt.AlignRight, '0')
        p.drawText(QRectF(l, t + h + 2, w, 16), Qt.AlignRight, f'episode {hist[-1]["episode"]}')
        for key, col, _ in SERIES:
            poly = QPolygonF([QPointF(l + w * (i / max(1, n - 1)), t + h - h * hh[key] / ymax)
                              for i, hh in enumerate(hist)])
            p.setPen(QPen(col, 1.6))
            p.drawPolyline(poly)
