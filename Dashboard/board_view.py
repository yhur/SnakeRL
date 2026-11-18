"""The original SnakeGame board, with the snake's vision rays drawn on top."""
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

import RLAgent.environment  # noqa: F401  (makes the SnakeGame folder importable)
from boards import GameBoard


class NoSnake:
    body = []


class BoardView(GameBoard):
    def __init__(self, cfg):
        super().__init__(x=cfg.grid_x, y=cfg.grid_y, block_size=26)
        self.setMinimumSize(260, 260)
        self.snake = NoSnake()
        self.game = None
        self.show_vision = True
        self.q_values = None                  # set while paused: [Q(forward), Q(right), Q(left)]

    def show_game(self, game):
        """Point the original GameBoard at this game's snake and apple."""
        self.game = game
        self.x, self.y = game.cfg.grid_x, game.cfg.grid_y
        self.snake = game.snake
        self.apple = game.board.apple
        self.update()

    def paintEvent(self, event):
        if self.game is None:
            return
        self.apple = self.game.board.apple
        super().paintEvent(event)            # the original game draws the board, snake and apple

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cell, ox, oy = self.board_geometry()

        def center_of(x, y):
            return QPointF(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell)

        if self.show_vision and self.game.cfg.state_type == 'simple':
            # the 'simple' state has no rays: outline the cells forward / right / left of the head
            for letter, cell_point in zip('FRL', self.game.cells_around_head()):
                color = QColor(255, 60, 60) if self.game.is_danger(cell_point) else QColor(0, 230, 90)
                painter.setPen(QPen(color, 2))
                painter.drawRect(QRectF(ox + cell_point.x * cell + 1, oy + cell_point.y * cell + 1,
                                        cell - 2, cell - 2))
        elif self.show_vision:
            hx, hy = self.game.snake.head.x, self.game.snake.head.y
            head = center_of(hx, hy)
            for dx, dy, wall, apple, body in self.game.ray_hits:
                painter.setPen(QPen(QColor(255, 255, 255, 60), 1))             # ray to the wall
                painter.drawLine(head, center_of(hx + dx * (wall - 0.5), hy + dy * (wall - 0.5)))
                if body:
                    painter.setPen(QPen(QColor(255, 160, 0), 2))               # sees its own body
                    painter.drawLine(head, center_of(hx + dx * body, hy + dy * body))
                if apple:
                    painter.setPen(QPen(QColor(0, 255, 80), 2.5))              # sees the apple
                    painter.drawLine(head, center_of(hx + dx * apple, hy + dy * apple))

        if self.q_values is not None:
            self.draw_q_values(painter, cell, ox, oy)

        font = QFont('Helvetica')
        font.setPixelSize(max(11, int(cell * 0.7)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(ox + 6, oy + 4, cell * self.x, cell), Qt.AlignLeft,
                         f'Score {self.game.score}   Steps {self.game.steps}')

    def draw_q_values(self, painter, cell, ox, oy):
        """Color each cell the snake can move into by its Q-value, and write the value in it.
        Green = the best move (biggest Q), red = the worst, yellow in between."""
        low, high = min(self.q_values), max(self.q_values)
        font = QFont('Helvetica')
        font.setPixelSize(max(9, int(cell * 0.4)))
        font.setBold(True)
        painter.setFont(font)
        # cells_around_head() is [forward, right, left]: the same order as the Q-values
        for q, point in zip(self.q_values, self.game.cells_around_head()):
            share = (q - low) / (high - low) if high > low else 1        # 0 = worst ... 1 = best
            box = QRectF(ox + point.x * cell, oy + point.y * cell, cell, cell)
            painter.fillRect(box, QColor.fromHsvF(share / 3, 0.9, 0.95, 0.8))   # hue: red -> yellow -> green
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(box, Qt.AlignCenter, f'{q:.1f}')

    def keyPressEvent(self, event):
        event.ignore()                        # keys are handled by the dashboard window
