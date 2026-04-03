from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@dataclass
class PPOConfig:
    gamma: float         = 0.99
    lam: float           = 0.95    # GAE lambda
    clip_ratio: float    = 0.2     # PPO clip epsilon
    learning_rate: float = 3e-4
    epochs: int          = 4       # optimisation epochs per update
    batch_size: int      = 64
    entropy_coef: float  = 0.01   # entropy bonus (encourages exploration)
    value_coef: float    = 0.5    # weight on value loss
    max_grad_norm: float = 0.5    # gradient clip norm


class PPOAgent:
    """
    PPO-Clip with a shared-backbone actor-critic.

    Architecture:  Input → Dense(128) → Dense(128)
                                             ├─► Actor  head: Dense(action_size, softmax)
                                             └─► Critic head: Dense(1, linear)

    Training is on-policy:
        1. Collect a full episode by calling act() + store() at each step.
        2. Call train(last_value) once after done=True.
        3. The buffer is cleared automatically after each train() call.
    """

    def __init__(self, state_size: int, action_size: int, cfg: Optional[PPOConfig] = None):
        self.state_size  = state_size
        self.action_size = action_size
        self.cfg         = cfg or PPOConfig()
        self.actor, self.critic = self._build_networks()
        self.optimizer = tf.keras.optimizers.Adam(self.cfg.learning_rate)
        self._buf: dict = {k: [] for k in ("s", "a", "r", "v", "lp", "d")}

    # network

    def _build_networks(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        inp = layers.Input(shape=(self.state_size,))
        x   = layers.Dense(128, activation="relu")(inp)
        x   = layers.Dense(128, activation="relu")(x)
        actor  = tf.keras.Model(inp, layers.Dense(self.action_size, activation="softmax")(x), name="actor")
        critic = tf.keras.Model(inp, layers.Dense(1)(x), name="critic")
        return actor, critic

    # action

    def act(self, state: np.ndarray, greedy: bool = False) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value). Pass all three to store()."""
        probs = self.actor(state, training=False).numpy()[0]
        val   = float(self.critic(state, training=False).numpy()[0, 0])
        a     = int(np.argmax(probs)) if greedy else int(np.random.choice(self.action_size, p=probs))
        return a, float(np.log(probs[a] + 1e-8)), val

    # buffer

    def store(self, state, action, reward, value, log_prob, done) -> None:
        self._buf["s"].append(state.flatten())
        self._buf["a"].append(int(action))
        self._buf["r"].append(float(reward))
        self._buf["v"].append(float(value))
        self._buf["lp"].append(float(log_prob))
        self._buf["d"].append(bool(done))

    # ── GAE ───────────────────────────────────────────────────────────────────

    def _gae(self, last_v: float) -> Tuple[np.ndarray, np.ndarray]:
        r  = np.array(self._buf["r"], dtype=np.float32)
        v  = np.array(self._buf["v"] + [last_v], dtype=np.float32)
        d  = np.array(self._buf["d"], dtype=np.float32)
        T  = len(r)
        adv = np.zeros(T, dtype=np.float32)
        g   = 0.0
        for t in reversed(range(T)):
            delta = r[t] + self.cfg.gamma * v[t + 1] * (1 - d[t]) - v[t]
            g     = delta + self.cfg.gamma * self.cfg.lam * (1 - d[t]) * g
            adv[t] = g
        return adv, adv + v[:-1]

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, last_value: float = 0.0) -> float:
        """Run PPO update on the stored trajectory, then clear the buffer."""
        adv, ret = self._gae(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S   = np.array(self._buf["s"],  dtype=np.float32)
        A   = np.array(self._buf["a"],  dtype=np.int32)
        OLP = np.array(self._buf["lp"], dtype=np.float32)
        R   = ret.astype(np.float32)
        ADV = adv.astype(np.float32)
        n   = len(S)
        total_loss = 0.0

        for _ in range(self.cfg.epochs):
            for b in [np.random.permutation(n)[s:s + self.cfg.batch_size]
                      for s in range(0, n, self.cfg.batch_size)]:
                sb, ab, ob, rb, ab2 = S[b], A[b], OLP[b], R[b], ADV[b]
                with tf.GradientTape() as tape:
                    probs  = self.actor(sb,  training=True)
                    vals   = self.critic(sb, training=True)[:, 0]
                    lp     = tf.math.log(tf.reduce_sum(probs * tf.one_hot(ab, self.action_size), 1) + 1e-8)
                    ratio  = tf.exp(lp - ob)
                    clip   = tf.clip_by_value(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
                    a_loss = -tf.reduce_mean(tf.minimum(ratio * ab2, clip * ab2))
                    v_loss = tf.reduce_mean(tf.square(vals - rb))
                    ent    = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), 1))
                    loss   = a_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent
                vs = self.actor.trainable_variables + self.critic.trainable_variables
                g, _ = tf.clip_by_global_norm(tape.gradient(loss, vs), self.cfg.max_grad_norm)
                self.optimizer.apply_gradients(zip(g, vs))
                total_loss += float(loss)

        for lst in self._buf.values():
            lst.clear()
        return total_loss

    #persistence 

    def save(self, path: str) -> None:
        self.actor.save(f"{path}_actor.keras")
        self.critic.save(f"{path}_critic.keras")

    def load(self, path: str) -> None:
        self.actor  = tf.keras.models.load_model(f"{path}_actor.keras")
        self.critic = tf.keras.models.load_model(f"{path}_critic.keras")