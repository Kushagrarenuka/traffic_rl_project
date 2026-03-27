from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class EnvConfig:
    max_steps: int = 200
    num_approaches: int = 8

    base_arrival_rate: float = 2.0
    peak_multiplier: float = 1.6
    service_rate: float = 5.0

    min_green: int = 5
    switch_penalty: float = 0.25

    w_total_queue: float = 1.0
    w_imbalance: float = 0.20
    w_delay: float = 0.15

    reward_scale: float = 0.01

    use_external_factors: bool = True
    normalize_state: bool = True
    max_queue_clip: float = 200.0


class TrafficEnvironment:
    """
    Single-intersection two-phase environment.

    Phase 0 serves lanes [0,1,2,3]
    Phase 1 serves lanes [4,5,6,7]

    Action:
        0 -> keep current phase
        1 -> request switch phase
    """

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
        demand_profile: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.demand_profile = demand_profile

        self.num_approaches = self.cfg.num_approaches
        self.action_size = 2

        self.phase = 0
        self.phase_timer = 0
        self.step_count = 0

        self.weather = 0.0
        self.event = 0.0
        self.queues = np.zeros(self.num_approaches, dtype=np.float32)

        extra = 4 if self.cfg.use_external_factors else 2
        self.state_size = self.num_approaches + extra

        self.cumulative_delay = 0.0
        self.total_switches = 0

    def reset(self) -> np.ndarray:
        self.phase = 0
        self.phase_timer = 0
        self.step_count = 0

        self.weather = float(self.rng.uniform(0.0, 1.0))
        self.event = float(self.rng.random() < 0.2) if self.cfg.use_external_factors else 0.0

        self.queues = self.rng.integers(
            low=5,
            high=15,
            size=self.num_approaches,
        ).astype(np.float32)

        self.cumulative_delay = 0.0
        self.total_switches = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, Dict[str, float]]:
        action = int(action)
        self.step_count += 1
        self.phase_timer += 1

        switched = 0
        if action == 1 and self.phase_timer >= self.cfg.min_green:
            self.phase = 1 - self.phase
            self.phase_timer = 0
            switched = 1
            self.total_switches += 1

        arrivals = self._sample_arrivals()
        self.queues += arrivals

        served = self._sample_service()
        self.queues = np.maximum(0.0, self.queues - served).astype(np.float32)
        self.queues = np.clip(self.queues, 0.0, self.cfg.max_queue_clip)

        step_delay = float(np.sum(self.queues))
        self.cumulative_delay += step_delay

        reward = self._compute_reward(switched=switched, step_delay=step_delay)
        done = self.step_count >= self.cfg.max_steps

        info = {
            "phase": float(self.phase),
            "phase_timer": float(self.phase_timer),
            "switched": float(switched),
            "total_queue": float(np.sum(self.queues)),
            "avg_queue": float(np.mean(self.queues)),
            "imbalance": float(np.sum(np.abs(self.queues - np.mean(self.queues)))),
            "step_delay": step_delay,
            "cumulative_delay": float(self.cumulative_delay),
            "weather": float(self.weather),
            "event": float(self.event),
            "switches_so_far": float(self.total_switches),
        }
        return self._get_state(), float(reward), bool(done), info

    def _get_state(self) -> np.ndarray:
        queue_state = self.queues.copy()
        if self.cfg.normalize_state:
            queue_state = queue_state / self.cfg.max_queue_clip

        phase_features = np.array(
            [
                float(self.phase),
                min(1.0, self.phase_timer / max(1.0, self.cfg.min_green)),
            ],
            dtype=np.float32,
        )

        if not self.cfg.use_external_factors:
            return np.concatenate([queue_state, phase_features], axis=0).astype(np.float32)

        external = np.array(
            [
                float(self.weather),
                float(self.event),
            ],
            dtype=np.float32,
        )
        return np.concatenate([queue_state, phase_features, external], axis=0).astype(np.float32)

    def _current_profile_multiplier(self) -> float:
        if self.demand_profile is None:
            return 1.0

        idx = min(self.step_count - 1, len(self.demand_profile) - 1)
        return float(self.demand_profile[idx])

    def _sample_arrivals(self) -> np.ndarray:
        profile_mult = self._current_profile_multiplier()
        event_mult = self.cfg.peak_multiplier if self.event > 0.5 else 1.0
        weather_mult = 1.0 + 0.5 * self.weather

        lam = self.cfg.base_arrival_rate * profile_mult * event_mult * weather_mult
        arrivals = self.rng.poisson(lam=lam, size=self.num_approaches).astype(np.float32)
        return arrivals

    def _sample_service(self) -> np.ndarray:
        served = np.zeros(self.num_approaches, dtype=np.float32)

        green_idx = np.arange(0, 4) if self.phase == 0 else np.arange(4, 8)
        weather_factor = 1.0 - 0.35 * self.weather
        mu = max(1.0, self.cfg.service_rate * weather_factor)

        served_amount = self.rng.poisson(lam=mu, size=green_idx.shape[0]).astype(np.float32)
        served[green_idx] = served_amount
        return served

    def _compute_reward(self, switched: int, step_delay: float) -> float:
        total_queue = float(np.sum(self.queues))
        imbalance = float(np.sum(np.abs(self.queues - np.mean(self.queues))))

        cost = (
            self.cfg.w_total_queue * total_queue
            + self.cfg.w_imbalance * imbalance
            + self.cfg.w_delay * step_delay
            + self.cfg.switch_penalty * float(switched)
        )
        return -self.cfg.reward_scale * cost
