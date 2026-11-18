"""Draws the snake's brain.

Wires:   blue = positive weight, red = negative weight, thicker = stronger.
Neurons: the brighter, the more strongly it fired for the current move.
"""
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

BACKGROUND = QColor(24, 26, 30)
BLUE = (70, 140, 255)
RED = (255, 70, 70)
WHITE = (235, 235, 235)
GREY = (110, 110, 110)
GREEN = (0, 230, 60)
CYAN = (0, 225, 255)


def blend(color1, color2, amount):
    """A color between color1 (amount 0) and color2 (amount 1)."""
    amount = min(max(amount, 0), 1)
    return QColor(*[int(a + (b - a) * amount) for a, b in zip(color1, color2)])


class NetworkView(QWidget):
    def __init__(self):
        super().__init__()
        self.brain = None
        self.input_labels = []
        self.output_labels = []
        self.setMinimumSize(420, 400)

    def show_brain(self, brain, input_labels, output_labels):
        self.brain = brain
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.update()

    def neuron_positions(self):
        """Screen position of every neuron: one column per layer."""
        sizes = self.brain.layer_sizes
        left, right, top, bottom = 70, 85, 34, 20
        width = self.width() - left - right
        height = self.height() - top - bottom
        radius = max(3, min(9, height / max(sizes) * 0.36))
        positions = []
        for layer, count in enumerate(sizes):
            x = left + width * layer / (len(sizes) - 1)
            gap = min(height / count, radius * 4.2)
            y_start = top + (height - gap * (count - 1)) / 2
            positions.append([QPointF(x, y_start + gap * i) for i in range(count)])
        return positions, radius

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        if self.brain is None:
            return
        positions, radius = self.neuron_positions()

        # ---- wires: one per weight -----------------------------------------
        for layer, W in enumerate(self.brain.weights):
            strongest = np.abs(W).max()
            wires = []
            for to in range(W.shape[0]):
                for frm in range(W.shape[1]):
                    wires.append((abs(W[to, frm]) / strongest, W[to, frm], frm, to))
            wires.sort()                            # weak wires first, so strong ones end up on top
            for strength, weight, frm, to in wires:
                color = QColor(*(BLUE if weight > 0 else RED))
                color.setAlphaF(0.1 + 0.9 * strength)
                painter.setPen(QPen(color, 0.4 + 2.6 * strength))
                painter.drawLine(positions[layer][frm], positions[layer + 1][to])

        # ---- neurons -------------------------------------------------------------
        last = len(positions) - 1
        chosen = int(np.argmax(self.brain.activations[-1]))
        for layer, column in enumerate(positions):
            values = self.brain.activations[layer]
            biggest = max(np.abs(values).max(), 1e-9)
            for i, point in enumerate(column):
                v = values[i] / biggest
                if layer == 0:
                    color = blend(WHITE, GREEN, v)                 # input: green when it sees something
                elif layer == last:
                    color = QColor(*GREEN) if i == chosen else blend(WHITE, (120, 200, 140), v)
                else:
                    color = blend(GREY, CYAN, v)                   # hidden: cyan when active
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                painter.setBrush(color)
                painter.drawEllipse(point, radius, radius)

        # ---- labels --------------------------------------------------------------
        font = QFont('Helvetica')
        font.setPixelSize(int(max(8, min(13, radius * 1.5))))
        painter.setFont(font)
        painter.setPen(QColor(210, 210, 210))
        for point, label in zip(positions[0], self.input_labels):
            painter.drawText(QRectF(point.x() - radius - 64, point.y() - 8, 60, 16),
                             Qt.AlignRight | Qt.AlignVCenter, label)
        for point, label in zip(positions[-1], self.output_labels):
            painter.drawText(QRectF(point.x() + radius + 6, point.y() - 8, 80, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.setPen(QColor(160, 160, 160))
        for layer, column in enumerate(positions):
            name = 'Input' if layer == 0 else 'Output' if layer == last else f'Hidden {layer}'
            painter.drawText(QRectF(column[0].x() - 50, 6, 100, 18), Qt.AlignCenter,
                             f'{name} ({len(column)})')
        painter.drawText(QRectF(8, self.height() - 18, self.width() - 16, 16), Qt.AlignLeft,
                         'wires: blue = positive, red = negative, thicker = stronger weight')
