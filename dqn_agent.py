from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.985

    replay_capacity: int = 50_000
    batch_size: int = 64
    train_start: int = 2_000

    target_update_every_steps: int = 1_000
    use_double_dqn: bool = True


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, cfg: Optional[DQNConfig] = None) -> None:
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.cfg = cfg or DQNConfig()

        self.memory: Deque[Transition] = deque(maxlen=self.cfg.replay_capacity)
        self.epsilon = self.cfg.epsilon_start
        self._train_step_counter = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(hard=True)

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(self.state_size,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss=tf.keras.losses.Huber(),
        )
        return model

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    def train_step(self) -> float | None:
        if len(self.memory) < self.cfg.train_start:
            return None

        batch = random.sample(self.memory, self.cfg.batch_size)

        states = np.vstack([b[0] for b in batch]).astype(np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch]).astype(np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        q_current = self.model.predict(states, verbose=0)
        target = q_current.copy()

        if self.cfg.use_double_dqn:
            next_q_online = self.model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q_online, axis=1)

            next_q_target = self.target_model.predict(next_states, verbose=0)
            next_values = next_q_target[np.arange(self.cfg.batch_size), next_actions]
        else:
            next_q_target = self.target_model.predict(next_states, verbose=0)
            next_values = np.max(next_q_target, axis=1)

        td_target = rewards + (1.0 - dones) * self.cfg.gamma * next_values
        target[np.arange(self.cfg.batch_size), actions] = td_target

        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = float(history.history["loss"][0])

        self._train_step_counter += 1
        if self._train_step_counter % self.cfg.target_update_every_steps == 0:
            self.update_target_network(hard=True)

        return loss

    def end_episode(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def update_target_network(self, hard: bool = True, tau: float = 1.0) -> None:
        if hard:
            self.target_model.set_weights(self.model.get_weights())
            return

        online = self.model.get_weights()
        target = self.target_model.get_weights()
        new_target = [tau * o + (1.0 - tau) * t for o, t in zip(online, target)]
        self.target_model.set_weights(new_target)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())