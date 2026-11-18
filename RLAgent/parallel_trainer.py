"""Headless only: several games at once, sharing one agent and one replay memory.

Each round:
    one network call picks a move for every game         (agent.choose_actions)
    every game makes its move, and it is remembered
    one training step, on a batch of batch_size x games  (agent.learn)

Asking the network about 8 states costs about the same as asking about 1, so this is several
times faster than Trainer. The network still learns from as many memories per move, but in
one big training step per round instead of one small step per move.
The dashboard uses Trainer: one game, easy to follow.
"""
from .environment import SnakeEnv
from .trainer import Trainer


class ParallelTrainer(Trainer):
    def __init__(self, cfg, agent=None, games=8):
        super().__init__(cfg, agent)                     # the agent and the progress (episode, history, ...)
        self.envs = [SnakeEnv(cfg) for _ in range(games)]
        self.states = [env.reset() for env in self.envs]
        self.returns = [0.0] * games                     # sum of rewards in each game
        self.losses = [[] for _ in range(games)]         # losses during each game

    @classmethod
    def from_trainer(cls, trainer, games):
        """The same agent and progress (e.g. from a checkpoint), now playing several games at once."""
        parallel = cls(trainer.cfg, trainer.agent, games)
        parallel.episode, parallel.history = trainer.episode, trainer.history
        parallel.best_score, parallel.best_episode = trainer.best_score, trainer.best_episode
        return parallel

    def step(self):
        """One move in every game, then learn. Returns True when at least one game ended."""
        actions, _ = self.agent.choose_actions(self.states)
        finished = []
        for i, env in enumerate(self.envs):
            next_state, reward, done = env.step(actions[i])
            self.agent.remember(self.states[i], actions[i], reward, next_state, done)
            self.returns[i] += reward
            self.states[i] = next_state
            if done:
                finished.append(i)

        loss = self.agent.learn(moves=len(self.envs))
        if loss is not None:
            for game_losses in self.losses:
                game_losses.append(loss)

        for i in finished:                               # record the finished games, start new ones
            self.record_game(self.envs[i], self.returns[i], self.losses[i])
            self.states[i] = self.envs[i].reset()
            self.returns[i], self.losses[i] = 0.0, []
        return bool(finished)
