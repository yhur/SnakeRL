"""The reinforcement-learning loop, one move at a time.

    for each episode (one game):
        state = env.reset()
        until the game ends:                                       <- step()
            action = agent.choose_action(state)                    epsilon-greedy
            next_state, reward, done = env.step(action)            the game moves the snake
            agent.remember(state, action, reward, next_state, done)
            agent.learn()                                          train on a random batch of memories
            state = next_state
        agent.end_episode()                                        explore a little less

The dashboard calls step() as fast as the chosen speed, or once per Space key while paused.
Headless mode calls it in a plain loop.
"""
from .environment import SnakeEnv
from .dqn_agent import DQNAgent

AVERAGE_OVER = 50     # episodes in the moving average


class Trainer:
    def __init__(self, cfg, agent=None):
        self.cfg = cfg
        self.env = SnakeEnv(cfg)
        self.agent = agent or DQNAgent(cfg)
        self.learning = True      # False: just watch it play its best moves, no exploring, no training

        self.episode = 1          # the game being played now
        self.history = []         # one entry per finished training episode, for the chart
        self.best_score = 0
        self.best_episode = 0
        self.previous = '-'       # how the last game ended
        self.last_move = None     # the move just made: state, action, reward, ... (for the dashboard).
        self.start_episode()      # It is kept when a new game starts, so the deadly move can be seen.

    def start_episode(self):
        self.state = self.env.reset()
        self.episode_return = 0.0       # sum of the rewards in this game
        self.episode_losses = []

    def step(self):
        """Make one move and learn from it. Returns True when that move ended the game."""
        state = self.state
        action, explored = self.agent.choose_action(state, explore=self.learning)
        next_state, reward, done = self.env.step(action)

        loss = None
        if self.learning:
            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.learn()
            if loss is not None:
                self.episode_losses.append(loss)

        self.episode_return += reward
        self.last_move = dict(state=state, action=action, explored=explored, reward=reward,
                              next_state=next_state, done=done, status=self.env.status, loss=loss)
        self.state = next_state
        if done:
            self.finish_episode()
        return done

    def finish_episode(self):
        self.record_game(self.env, self.episode_return, self.episode_losses)
        self.start_episode()

    def record_game(self, env, total_reward, losses):
        """Count a finished game: history for the chart, best score, and a little less exploring."""
        self.previous = f'{env.score} apples, {env.status}'
        if not self.learning:
            return                  # while only watching, games are not counted as training episodes
        if env.score > self.best_score or not self.best_episode:
            self.best_score, self.best_episode = env.score, self.episode
        recent = [h['score'] for h in self.history[-(AVERAGE_OVER - 1):]] + [env.score]
        self.history.append(dict(
            episode=self.episode, score=env.score, steps=env.steps, status=env.status,
            avg_score=sum(recent) / len(recent), epsilon=self.agent.epsilon, total_reward=total_reward,
            loss=sum(losses) / len(losses) if losses else None))
        self.agent.end_episode()
        self.episode += 1

    def use_settings(self, cfg):
        """New settings, same agent: start a fresh game with them."""
        self.cfg = cfg
        self.agent.apply_settings(cfg)
        self.env = SnakeEnv(cfg)
        self.start_episode()
