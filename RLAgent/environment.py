"""The environment: the Snake game seen through the eyes of a reinforcement-learning agent.

The agent and the environment talk in a loop:

        state  ──►  agent picks an action  ──►  environment moves the snake
          ▲                                                   │
          └─────────────  next state, reward, done  ◄─────────┘

The game itself is the original, unmodified code in ../SnakeGame (snake.Snake and boards.Board).
This file only adds what RL needs: a state (a list of numbers) and a reward.

Two kinds of state (setting 'state_type'):
  vision   the same inputs as SnakeGenetics: the snake looks along 8 (or 4) rays and sees
           the wall, the apple and its own body; plus the head and the tail direction. 32 numbers.
  simple   14 yes/no facts: a wall forward/right/left?, the moving direction,
           the snake's own body forward/right/left?, which side the apple is.
"""
import os
import sys
import numpy as np

# Make the original game importable.
GAME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SnakeGame')
sys.path.insert(0, GAME_DIR)
from snake import Snake, Direction, Point   # noqa: E402
from boards import Board                    # noqa: E402

# The game's own move encoding: [forward, right, left]. Action i means ACTIONS[i].
ACTIONS = [Snake.action_f, Snake.action_r, Snake.action_l]
ACTION_NAMES = ['Forward', 'Right', 'Left']

# The game lists the directions clockwise: RIGHT, DOWN, LEFT, UP.
# So "turn right" is the next one in this list, and "turn left" the previous one.
CLOCKWISE = list(Direction)

# Vision: the directions the snake looks in, as (dx, dy). y grows downward on screen.
RAYS_8 = {'U': (0, -1), 'UR': (1, -1), 'R': (1, 0), 'DR': (1, 1),
          'D': (0, 1), 'DL': (-1, 1), 'L': (-1, 0), 'UL': (-1, -1)}
RAYS_4 = {'U': (0, -1), 'R': (1, 0), 'D': (0, 1), 'L': (-1, 0)}
UDLR = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

SIMPLE_LABELS = ['Wall F', 'Wall R', 'Wall L',
                 'Moving R', 'Moving D', 'Moving L', 'Moving U',
                 'Body F', 'Body R', 'Body L',
                 'Apple L', 'Apple R', 'Apple U', 'Apple D']


def rays_of(cfg):
    return RAYS_8 if cfg.vision_directions == 8 else RAYS_4


def state_labels(cfg):
    """What each state number means. Vision: 'U-W' = wall looking up, 'U-A' = apple, 'U-S' = self."""
    if cfg.state_type == 'simple':
        return SIMPLE_LABELS
    labels = []
    for name in rays_of(cfg):
        labels += [name + '-W', name + '-A', name + '-S']
    return labels + ['HU', 'HD', 'HL', 'HR', 'TU', 'TD', 'TL', 'TR']


class SnakeEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.board = Board(cfg.grid_x, cfg.grid_y)
        self.snake = Snake(cfg.grid_x, cfg.grid_y)
        self.board.snake = self.snake
        self.ray_hits = []               # vision: what each ray saw, for drawing
        self.reset()

    @property
    def score(self):
        return self.snake.score

    def reset(self):
        """Start a new game and return the first state."""
        self.board.reset()               # the original game: new snake, new apple
        self.steps = 0
        self.status = 'ok'               # 'ok', 'apple', or how it ended: 'wall', 'body', 'starved', 'won'
        self.done = False
        return self.get_state()

    def get_state(self):
        state = self.simple_state() if self.cfg.state_type == 'simple' else self.vision_state()
        return np.array(state, dtype=np.float32)

    # ---- state 'vision': the same inputs as SnakeGenetics -----------------
    def vision_state(self):
        """For every ray: 1/distance to the wall, whether the apple is on the ray, whether the
        snake's own body is on the ray. Then the head direction and the tail direction,
        each as 4 numbers where only one is 1 (up, down, left, right)."""
        head_x, head_y = int(self.snake.head.x), int(self.snake.head.y)
        body = [(int(p.x), int(p.y)) for p in self.snake.body[1:]]
        apple = (int(self.board.apple.x), int(self.board.apple.y))
        binary = self.cfg.apple_self_vision == 'binary'

        state = []
        self.ray_hits = []
        for dx, dy in rays_of(self.cfg).values():
            # walk along the ray until we leave the board
            x, y, distance = head_x + dx, head_y + dy, 1
            apple_distance, body_distance = 0, 0
            while 0 <= x < self.cfg.grid_x and 0 <= y < self.cfg.grid_y:
                if apple_distance == 0 and (x, y) == apple:
                    apple_distance = distance
                if body_distance == 0 and (x, y) in body:
                    body_distance = distance
                x, y, distance = x + dx, y + dy, distance + 1
            self.ray_hits.append((dx, dy, distance, apple_distance, body_distance))

            state.append(1 / distance)                       # wall: the closer, the bigger
            if binary:
                state.append(1 if apple_distance else 0)
                state.append(1 if body_distance else 0)
            else:
                state.append(1 / apple_distance if apple_distance else 0)
                state.append(1 / body_distance if body_distance else 0)

        for d in UDLR:                                       # which way the head is going
            state.append(1 if self.snake.direction == d else 0)
        tail, before_tail = self.snake.body[-1], self.snake.body[-2]
        tail_direction = (before_tail.x - tail.x, before_tail.y - tail.y)
        for d in UDLR:                                       # which way the tail is going
            state.append(1 if tail_direction == d.value else 0)
        return state

    # ---- state 'simple': 14 yes/no facts ------------------------------------
    def cells_around_head(self):
        """The cells straight ahead, to the right and to the left of the head."""
        i = CLOCKWISE.index(self.snake.direction)
        cells = []
        for direction in (CLOCKWISE[i], CLOCKWISE[(i + 1) % 4], CLOCKWISE[(i - 1) % 4]):
            dx, dy = direction.value
            cells.append(Point(self.snake.head.x + dx, self.snake.head.y + dy))
        return cells                     # [forward, right, left]

    def is_wall(self, cell):
        """Is this cell outside the board?"""
        return not (0 <= cell.x < self.cfg.grid_x and 0 <= cell.y < self.cfg.grid_y)

    def is_body(self, cell):
        """Is the snake's own body in this cell?"""
        return cell in self.snake.body

    def is_danger(self, cell):
        """Would moving into this cell end the game? (a wall, or the snake's own body)"""
        return self.is_wall(cell) or self.is_body(cell)

    def simple_state(self):
        head, apple, direction = self.snake.head, self.board.apple, self.snake.direction
        cells = self.cells_around_head()                                      # forward, right, left
        state = [self.is_wall(cell) for cell in cells]                        # wall F, R, L
        state += [direction == Direction.RIGHT, direction == Direction.DOWN,  # moving R, D, L, U
                  direction == Direction.LEFT, direction == Direction.UP]
        state += [self.is_body(cell) for cell in cells]                       # body F, R, L
        state += [apple.x < head.x, apple.x > head.x,                         # apple L, R, U, D
                  apple.y < head.y, apple.y > head.y]
        return state

    # ---- one move ---------------------------------------------------------
    def step(self, action):
        """Do action 0, 1 or 2 (forward, right, left). Returns (next_state, reward, done)."""
        self.status = self.snake.moveTo(ACTIONS[action])
        self.steps += 1

        if self.status == 'apple':
            reward = self.cfg.reward_apple
            if len(self.snake.body) == self.cfg.grid_x * self.cfg.grid_y:
                self.status, self.done = 'won', True          # no room left for an apple
            else:
                self.board.generate_apple(self.snake)
        elif self.status == 'ok':
            reward = self.cfg.reward_step
        else:                                                 # 'wall', 'body' or 'starved'
            reward = self.cfg.reward_death
            self.done = True

        return self.get_state(), reward, self.done


def describe_state(state, cfg):
    """The state in words, e.g. 'wall: R  |  moving: U  |  body: -  |  apple: L U'."""
    def names(values, letters):
        return ' '.join(letter for letter, value in zip(letters, values) if value) or '-'

    if cfg.state_type == 'simple':
        return f'wall: {names(state[0:3], "FRL")}  |  moving: {names(state[3:7], "RDLU")}  |  ' \
               f'body: {names(state[7:10], "FRL")}  |  apple: {names(state[10:14], "LRUD")}'
    rays = list(rays_of(cfg))
    n = len(rays) * 3
    return f'apple seen: {names(state[1:n:3], rays)}  |  body seen: {names(state[2:n:3], rays)}  |  ' \
           f'moving: {names(state[n:n + 4], "UDLR")}  |  tail: {names(state[n + 4:n + 8], "UDLR")}'
