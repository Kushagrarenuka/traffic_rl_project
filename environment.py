from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Single-intersection environment 


@dataclass
class EnvConfig:
    max_steps: int          = 200
    num_approaches: int     = 8
    base_arrival_rate: float = 2.0
    peak_multiplier: float  = 1.6
    service_rate: float     = 5.0
    min_green: int          = 5
    switch_penalty: float   = 0.25
    w_total_queue: float    = 1.0
    w_imbalance: float      = 0.20
    w_delay: float          = 0.15
    reward_scale: float     = 0.01
    use_external_factors: bool = True
    normalize_state: bool   = True
    max_queue_clip: float   = 200.0


class TrafficEnvironment:
    """
    Original single-intersection two-phase environment.
    Kept unchanged for backward-compatible single-agent runs.

    Phase 0 → lanes [0,1,2,3]   Phase 1 → lanes [4,5,6,7]
    Action: 0 = keep phase, 1 = switch phase
    """

    def __init__(self, cfg=None, seed=None, demand_profile=None):
        self.cfg            = cfg or EnvConfig()
        self.rng            = np.random.default_rng(seed)
        self.demand_profile = demand_profile
        self.num_approaches = self.cfg.num_approaches
        self.action_size    = 2
        extra               = 4 if self.cfg.use_external_factors else 2
        self.state_size     = self.num_approaches + extra
        self._reset_state()

    def _reset_state(self):
        self.phase            = 0
        self.phase_timer      = 0
        self.step_count       = 0
        self.weather          = 0.0
        self.event            = 0.0
        self.queues           = np.zeros(self.num_approaches, dtype=np.float32)
        self.cumulative_delay = 0.0
        self.total_switches   = 0

    def reset(self) -> np.ndarray:
        self._reset_state()
        self.weather = float(self.rng.uniform(0.0, 1.0))
        self.event   = float(self.rng.random() < 0.2) if self.cfg.use_external_factors else 0.0
        self.queues  = self.rng.integers(5, 15, size=self.num_approaches).astype(np.float32)
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count  += 1
        self.phase_timer += 1
        switched = 0
        if action == 1 and self.phase_timer >= self.cfg.min_green:
            self.phase       = 1 - self.phase
            self.phase_timer = 0
            switched         = 1
            self.total_switches += 1

        self.queues += self._sample_arrivals()
        self.queues  = np.clip(
            np.maximum(0.0, self.queues - self._sample_service()),
            0.0, self.cfg.max_queue_clip,
        ).astype(np.float32)

        step_delay             = float(np.sum(self.queues))
        self.cumulative_delay += step_delay
        reward = self._compute_reward(switched, step_delay)
        done   = self.step_count >= self.cfg.max_steps
        info   = dict(
            phase=float(self.phase), phase_timer=float(self.phase_timer),
            switched=float(switched), total_queue=float(np.sum(self.queues)),
            avg_queue=float(np.mean(self.queues)),
            imbalance=float(np.sum(np.abs(self.queues - np.mean(self.queues)))),
            step_delay=step_delay, cumulative_delay=float(self.cumulative_delay),
            weather=float(self.weather), event=float(self.event),
            switches_so_far=float(self.total_switches),
        )
        return self._get_state(), float(reward), bool(done), info

    def _get_state(self) -> np.ndarray:
        q = self.queues.copy()
        if self.cfg.normalize_state:
            q /= self.cfg.max_queue_clip
        pf  = np.array([float(self.phase),
                        min(1.0, self.phase_timer / max(1.0, self.cfg.min_green))],
                       dtype=np.float32)
        ext = np.array([float(self.weather), float(self.event)], dtype=np.float32)
        parts = [q, pf] + ([ext] if self.cfg.use_external_factors else [])
        return np.concatenate(parts).astype(np.float32)

    def _profile_mult(self) -> float:
        if self.demand_profile is None:
            return 1.0
        return float(self.demand_profile[min(self.step_count - 1, len(self.demand_profile) - 1)])

    def _sample_arrivals(self) -> np.ndarray:
        lam = (self.cfg.base_arrival_rate * self._profile_mult()
               * (self.cfg.peak_multiplier if self.event > 0.5 else 1.0)
               * (1.0 + 0.5 * self.weather))
        return self.rng.poisson(lam=lam, size=self.num_approaches).astype(np.float32)

    def _sample_service(self) -> np.ndarray:
        served = np.zeros(self.num_approaches, dtype=np.float32)
        idx    = np.arange(0, 4) if self.phase == 0 else np.arange(4, 8)
        mu     = max(1.0, self.cfg.service_rate * (1.0 - 0.35 * self.weather))
        served[idx] = self.rng.poisson(lam=mu, size=4).astype(np.float32)
        return served

    def _compute_reward(self, switched: int, step_delay: float) -> float:
        cost = (self.cfg.w_total_queue * float(np.sum(self.queues))
                + self.cfg.w_imbalance * float(np.sum(np.abs(self.queues - np.mean(self.queues))))
                + self.cfg.w_delay    * step_delay
                + self.cfg.switch_penalty * float(switched))
        return -self.cfg.reward_scale * cost


# Multi-intersection environment


@dataclass
class MultiEnvConfig:
    """
    PROPOSAL: "A scalable multi-agent coordination system … multiple traffic
    lights and intersections work together to optimize city-wide traffic flow."
    """
    n_intersections: int    = 2        # set 2–4
    max_steps: int          = 200
    num_approaches: int     = 8
    base_arrival_rate: float = 2.0
    peak_multiplier: float  = 1.6
    service_rate: float     = 5.0
    min_green: int          = 5
    switch_penalty: float   = 0.25
    w_total_queue: float    = 1.0
    w_imbalance: float      = 0.20
    w_delay: float          = 0.15
    # PROPOSAL: "focusing on fuel efficiency and emission reduction"
    w_co2: float            = 0.10
    co2_scale: float        = 5.0     # brings CO2 cost to same order as queue costs
    reward_scale: float     = 0.01
    use_external_factors: bool = True
    normalize_state: bool   = True
    max_queue_clip: float   = 200.0
    spillback_fraction: float = 0.05  # overflow fraction that spills to next intersection


class MultiIntersectionEnvironment:
    """
    N intersections in a linear chain:  0 ──► 1 ──► 2 ──► … ──► N-1

    Additions over TrafficEnvironment
    ----------------------------------
    1. Multiple intersections with inter-connected spillback flow.
    2. Live weather severity injected from the OpenWeather API
       (falls back to synthetic if no key is set).
    3. CO2/fuel cost term in the reward function.
    4. Returns lists of states / rewards / infos — one entry per intersection.

    State per intersection (12-D with external factors):
        [q0..q7 normalised, phase, phase_timer_norm, weather, event]

    Action per intersection:
        0 = keep current phase    1 = request switch
    """

    def __init__(
        self,
        cfg: Optional[MultiEnvConfig] = None,
        seed: Optional[int] = None,
        demand_profile: Optional[np.ndarray] = None,
        weather_severity: float = 0.0,   # pass return value of get_weather_severity()
    ) -> None:
        self.cfg              = cfg or MultiEnvConfig()
        self.n                = self.cfg.n_intersections
        self.rng              = np.random.default_rng(seed)
        self.demand_profile   = demand_profile
        self.weather_severity = weather_severity   # 0.0 → use synthetic each episode

        self.num_approaches = self.cfg.num_approaches
        self.action_size    = 2
        extra               = 4 if self.cfg.use_external_factors else 2
        self.state_size     = self.num_approaches + extra   # 12

        # All per-intersection state — properly initialised in reset()
        self.phases:            List[int]        = [0]   * self.n
        self.phase_timers:      List[int]        = [0]   * self.n
        self.queues:            List[np.ndarray] = [np.zeros(self.num_approaches, dtype=np.float32)] * self.n
        self.step_count:        int              = 0
        self.weather:           float            = 0.0
        self.event:             float            = 0.0
        self.cumulative_delays: List[float]      = [0.0] * self.n
        self.total_switches:    List[int]        = [0]   * self.n
        self.total_co2:         List[float]      = [0.0] * self.n

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        self.step_count       = 0
        self.phases           = [0]   * self.n
        self.phase_timers     = [0]   * self.n
        self.cumulative_delays = [0.0] * self.n
        self.total_switches   = [0]   * self.n
        self.total_co2        = [0.0] * self.n

        # Use live API value when available; else synthetic random each episode
        self.weather = (
            float(np.clip(self.weather_severity, 0.0, 1.0))
            if self.weather_severity > 0.0
            else float(self.rng.uniform(0.0, 1.0))
        )
        self.event = float(self.rng.random() < 0.2) if self.cfg.use_external_factors else 0.0
        self.queues = [
            self.rng.integers(5, 15, size=self.num_approaches).astype(np.float32)
            for _ in range(self.n)
        ]
        return [self._get_state(i) for i in range(self.n)]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, List[Dict]]:
        """
        Apply one action per intersection simultaneously.

        Returns
        -------
        states  : per-intersection state vectors
        rewards : per-intersection scalar rewards (includes CO2 cost)
        done    : True when max_steps reached
        infos   : per-intersection diagnostic dicts
        """
        self.step_count += 1
        switched = self._update_phases(actions)
        self._update_queues()
        rewards, infos = self._compute_rewards_infos(switched)
        done   = self.step_count >= self.cfg.max_steps
        states = [self._get_state(i) for i in range(self.n)]
        return states, rewards, done, infos

    # ── internals ─────────────────────────────────────────────────────────────

    def _update_phases(self, actions: List[int]) -> List[int]:
        sw = []
        for i in range(self.n):
            self.phase_timers[i] += 1
            s = 0
            if actions[i] == 1 and self.phase_timers[i] >= self.cfg.min_green:
                self.phases[i]       = 1 - self.phases[i]
                self.phase_timers[i] = 0
                s                    = 1
                self.total_switches[i] += 1
            sw.append(s)
        return sw

    def _update_queues(self) -> None:
        arrivals = self._sample_arrivals()
        for i in range(self.n):
            self.queues[i] += arrivals
            # Spillback: overflow from the upstream intersection
            if i > 0:
                overflow = np.maximum(0.0, self.queues[i - 1] - self.cfg.max_queue_clip)
                self.queues[i] += overflow * self.cfg.spillback_fraction
            self.queues[i] -= self._sample_service(i)
            self.queues[i]  = np.clip(self.queues[i], 0.0, self.cfg.max_queue_clip).astype(np.float32)

    def _compute_rewards_infos(self, switched: List[int]) -> Tuple[List[float], List[Dict]]:
        rewards, infos = [], []
        for i in range(self.n):
            tq        = float(np.sum(self.queues[i]))
            imbal     = float(np.sum(np.abs(self.queues[i] - np.mean(self.queues[i]))))
            step_del  = tq
            self.cumulative_delays[i] += step_del

            # CO2: idling vehicles proportional to queue length
            # PROPOSAL: "minimize … fuel efficiency and CO2 emissions during high-volume periods"
            co2_step = step_del * self.cfg.co2_scale * 0.001
            self.total_co2[i] += co2_step

            cost = (
                self.cfg.w_total_queue  * tq
                + self.cfg.w_imbalance  * imbal
                + self.cfg.w_delay      * step_del
                + self.cfg.switch_penalty * float(switched[i])
                + self.cfg.w_co2        * co2_step
            )
            rewards.append(-self.cfg.reward_scale * cost)
            infos.append(dict(
                intersection=i, phase=float(self.phases[i]),
                phase_timer=float(self.phase_timers[i]), switched=float(switched[i]),
                total_queue=tq, avg_queue=float(np.mean(self.queues[i])),
                imbalance=imbal, step_delay=step_del,
                cumulative_delay=float(self.cumulative_delays[i]),
                co2_step=co2_step, total_co2=float(self.total_co2[i]),
                weather=float(self.weather), event=float(self.event),
                switches_so_far=float(self.total_switches[i]),
            ))
        return rewards, infos

    def _get_state(self, i: int) -> np.ndarray:
        q  = self.queues[i].copy()
        if self.cfg.normalize_state:
            q /= self.cfg.max_queue_clip
        pf  = np.array([float(self.phases[i]),
                        min(1.0, self.phase_timers[i] / max(1.0, float(self.cfg.min_green)))],
                       dtype=np.float32)
        ext = np.array([float(self.weather), float(self.event)], dtype=np.float32)
        parts = [q, pf] + ([ext] if self.cfg.use_external_factors else [])
        return np.concatenate(parts).astype(np.float32)

    def _profile_mult(self) -> float:
        if self.demand_profile is None:
            return 1.0
        return float(self.demand_profile[min(self.step_count - 1, len(self.demand_profile) - 1)])

    def _sample_arrivals(self) -> np.ndarray:
        lam = (self.cfg.base_arrival_rate * self._profile_mult()
               * (self.cfg.peak_multiplier if self.event > 0.5 else 1.0)
               * (1.0 + 0.5 * self.weather))
        return self.rng.poisson(lam=lam, size=self.num_approaches).astype(np.float32)

    def _sample_service(self, i: int) -> np.ndarray:
        served  = np.zeros(self.num_approaches, dtype=np.float32)
        idx     = np.arange(0, 4) if self.phases[i] == 0 else np.arange(4, 8)
        mu      = max(1.0, self.cfg.service_rate * (1.0 - 0.35 * self.weather))
        served[idx] = self.rng.poisson(lam=mu, size=4).astype(np.float32)
        return served