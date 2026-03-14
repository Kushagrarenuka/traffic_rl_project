from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EnvConfig:
    max_steps: int = 200

    # Arrivals
    base_arrival_rate: float = 2.0      # mean arrivals per lane per step
    peak_multiplier: float = 1.5

    # Service
    service_rate: float = 6.0           # mean vehicles served per green approach per step
    switch_penalty: float = 1.0         # discourages thrashing

    # Reward weights
    w_total_queue: float = 1.0
    w_imbalance: float = 0.25

    # External factors
    use_external_factors: bool = True


class TrafficEnvironment:
    """
    Two-phase intersection:
    - Phase 0 serves approaches [0,1,2,3]
    - Phase 1 serves approaches [4,5,6,7]
    """

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.num_approaches = 8
        self.action_size = 2

        self.phase = 0
        self.step_count = 0

        self.weather = 0.0
        self.event = 0.0

        self.queues = np.zeros(self.num_approaches, dtype=np.float32)

        self.state_size = self.num_approaches + (2 if self.cfg.use_external_factors else 0)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.phase = 0

        # Start with moderate queues
        self.queues = self.rng.integers(low=5, high=15, size=self.num_approaches).astype(np.float32)

        if self.cfg.use_external_factors:
            self.weather = float(self.rng.uniform(0.0, 1.0))  # 0 clear, 1 severe
            self.event = float(self.rng.random() < 0.2)       # 20% chance of event
        else:
            self.weather, self.event = 0.0, 0.0

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        action = int(action)
        self.step_count += 1

        switched = 0
        if action == 1:
            self.phase = 1 - self.phase
            switched = 1

        arrivals = self._sample_arrivals()
        self.queues += arrivals

        served = self._sample_service()
        self.queues = np.maximum(0.0, self.queues - served)

        reward = self._compute_reward(switched=switched)
        done = self.step_count >= self.cfg.max_steps

        info = {
            "phase": self.phase,
            "total_queue": float(np.sum(self.queues)),
            "weather": float(self.weather),
            "event": float(self.event),
        }
        return self._get_state(), float(reward), bool(done), info

    def _get_state(self) -> np.ndarray:
        if not self.cfg.use_external_factors:
            return self.queues.copy()

        return np.concatenate([self.queues, np.array([self.weather, self.event], dtype=np.float32)], axis=0)

    def _sample_arrivals(self) -> np.ndarray:
        # External factors increase arrivals and reduce service realism
        peak = self.cfg.peak_multiplier if self.event > 0.5 else 1.0
        lam = self.cfg.base_arrival_rate * peak * (1.0 + 0.5 * self.weather)
        return self.rng.poisson(lam=lam, size=self.num_approaches).astype(np.float32)

    def _sample_service(self) -> np.ndarray:
        served = np.zeros(self.num_approaches, dtype=np.float32)

        green_idx = np.arange(0, 4) if self.phase == 0 else np.arange(4, 8)

        # Bad weather reduces service
        mu = self.cfg.service_rate * (1.0 - 0.4 * self.weather)
        mu = max(1.0, mu)

        # Poisson service per green approach
        served_amount = self.rng.poisson(lam=mu, size=green_idx.shape[0]).astype(np.float32)
        served[green_idx] = served_amount
        return served

    def _compute_reward(self, switched: int) -> float:
        total_queue = float(np.sum(self.queues))
        imbalance = float(np.sum(np.abs(self.queues - np.mean(self.queues))))

        # Reward is negative cost (we want low queues + balanced queues)
        cost = self.cfg.w_total_queue * total_queue + self.cfg.w_imbalance * imbalance
        cost += self.cfg.switch_penalty * float(switched)
        return -cost