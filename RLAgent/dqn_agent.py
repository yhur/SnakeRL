"""The agent: a Deep Q-Network (DQN) built with TensorFlow / Keras.

Q(state, action) = "how much reward do I expect in the future if I make this move now?"

The network takes the state numbers (32 for Vision, 14 for Simple) and outputs 3 Q-values,
one per move (forward, right, left). The agent plays the move with the biggest Q-value,
except when it explores: then it plays a random move (epsilon-greedy).

It learns with the Bellman equation. After a move we know the reward we got and the
state we landed in, so a better guess for Q(state, action) is:

        target = reward + gamma * max Q(next_state, any move)      (just reward if the game ended)

Training nudges the network's Q(state, action) toward this target.
"""
import random
from types import SimpleNamespace
import numpy as np
import keras

from .replay_memory import ReplayMemory


def build_network(cfg):
    """state -> hidden layers -> Q-values (3)."""
    inputs = keras.Input(shape=(cfg.num_inputs,), name='state')
    x = inputs
    for i, neurons in enumerate(cfg.hidden_layers):
        x = keras.layers.Dense(neurons, activation=cfg.hidden_activation, name=f'hidden_{i + 1}')(x)
    # Q-values can be any number (negative too), so the output layer has no activation.
    q_values = keras.layers.Dense(cfg.num_outputs, activation='linear', name='q_values')(x)

    model = keras.Model(inputs, q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss=cfg.loss)
    return model


class DQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.seed:
            keras.utils.set_random_seed(cfg.seed)     # Python, NumPy and TensorFlow
        self.model = build_network(cfg)               # the network that learns and plays
        self.target_model = build_network(cfg)        # a frozen copy, used only for the targets
        self.target_model.set_weights(self.model.get_weights())
        self.memory = ReplayMemory(cfg.memory_size)

        self.epsilon = cfg.epsilon_start              # chance of a random move
        self.moves = 0                                # moves made while learning
        self.train_steps = 0                          # times the network was trained
        self.target_updates = 0                       # times the target network was refreshed
        self.last_loss = None

        # Same layers as `model`, but returns every layer's output. Only for the dashboard.
        dense_layers = [layer for layer in self.model.layers if isinstance(layer, keras.layers.Dense)]
        self.inspect_model = keras.Model(self.model.inputs, [layer.output for layer in dense_layers])

    # ---- acting -------------------------------------------------------------
    def q_values(self, state):
        """The network's Q-values for one state: [Q(forward), Q(right), Q(left)]."""
        return self.model.predict_on_batch(np.array([state]))[0]

    def choose_action(self, state, explore=True):
        """Epsilon-greedy. Returns (action, explored): explored is True for a random move."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.cfg.num_outputs), True
        return int(np.argmax(self.q_values(state))), False

    def choose_actions(self, states, explore=True):
        """choose_action() for several games at once, with one network call for all their states.
        Returns (actions, explored) as two lists. Used by ParallelTrainer."""
        all_q = self.model.predict_on_batch(np.array(states))
        actions, explored = [], []
        for q in all_q:
            if explore and random.random() < self.epsilon:
                actions.append(random.randrange(self.cfg.num_outputs))
                explored.append(True)
            else:
                actions.append(int(np.argmax(q)))
                explored.append(False)
        return actions, explored

    # ---- learning -----------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self, moves=1):
        """Called after every move. Trains on a random batch of memories when it is time.

        ParallelTrainer calls it once per round, with moves = the number of games. The batch then
        grows with it, so the network still learns from as many memories per move.
        Returns the loss, or None when it did not train."""
        cfg = self.cfg
        before = self.moves
        self.moves += moves
        trainings_due = self.moves // cfg.train_every - before // cfg.train_every   # usually 0 or 1
        batch_size = cfg.batch_size * trainings_due
        loss = None
        if trainings_due and len(self.memory) >= batch_size:
            loss = self.train_on_batch(*self.memory.sample(batch_size))
        if cfg.target_update and self.moves // cfg.target_update > before // cfg.target_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_updates += 1
        return loss

    def train_on_batch(self, states, actions, rewards, next_states, dones):
        """One step of Q-learning on a batch of memories."""
        # 1. What the network predicts now, for every move.
        targets = self.model.predict_on_batch(states)

        # 2. The best Q-value of the state after each move. The target network is a copy of
        #    the network from a while ago: chasing a target that moves with every training
        #    step makes learning unstable.
        target_net = self.target_model if self.cfg.target_update else self.model
        next_q = target_net.predict_on_batch(next_states)

        # 3. Bellman: target = reward + gamma * best future Q. No future when the game ended.
        future = self.cfg.gamma * next_q.max(axis=1) * (1 - dones)
        rows = np.arange(len(actions))
        targets[rows, actions] = rewards + future   # only the move actually made gets a new target

        # 4. Move the network's predictions toward the targets.
        self.last_loss = float(self.model.train_on_batch(states, targets))
        self.train_steps += 1
        return self.last_loss

    def end_episode(self):
        """After each game, explore a little less."""
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    # ---- for the dashboard --------------------------------------------------
    def best_future_q(self, next_state):
        """max Q(next_state, any move), as train_on_batch() computes it. To show one Bellman target."""
        target_net = self.target_model if self.cfg.target_update else self.model
        return float(target_net.predict_on_batch(np.array([next_state]))[0].max())

    def layer_values(self, state):
        """Every layer's values for one state: [inputs, hidden 1, ..., Q-values]."""
        outputs = self.inspect_model.predict_on_batch(np.array([state]))
        if not isinstance(outputs, (list, tuple)):    # a network without hidden layers
            outputs = [outputs]
        return [np.array(state)] + [out[0] for out in outputs]

    def brain(self, state):
        """The network in the form the dashboard's NetworkView (from SnakeGenetics) draws:
        layer sizes, weights[i][to, from] and every layer's values for this state.
        (Keras stores each weight matrix the other way round, [from, to], hence .T)"""
        return SimpleNamespace(layer_sizes=self.cfg.layer_sizes,
                               weights=[kernel.T for kernel in self.model.get_weights()[0::2]],
                               activations=self.layer_values(state))

    def apply_settings(self, cfg):
        """Use new settings but keep what was learned (same network shape)."""
        self.cfg = cfg
        self.model.optimizer.learning_rate.assign(cfg.learning_rate)
        if self.memory.capacity != cfg.memory_size:
            self.memory.resize(cfg.memory_size)
        self.epsilon = max(cfg.epsilon_min, min(self.epsilon, cfg.epsilon_start))
