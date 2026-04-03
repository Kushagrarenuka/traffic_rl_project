from __future__ import annotations
import random
import numpy as np


class FixedTimeController:
    """Switch phase every fixed cycle length, respecting min-green."""
    def __init__(self, cycle_length: int = 10) -> None:
        self.cycle_length = cycle_length

    def act(self, phase_timer: int, queues=None) -> int:
        return 1 if phase_timer >= self.cycle_length else 0


class RandomController:
    """
    Acts uniformly at random — useful as a lower-bound baseline.
    PROPOSAL: "Compare mean std over multiple random seeds" — having a random
    baseline makes the comparison table more complete.
    """
    def act(self, phase_timer: int, queues=None) -> int:
        return random.randint(0, 1)


class MaxPressureController:
    """
    Actuated heuristic: switch when the queue pressure difference between
    the two phases exceeds a threshold.  Approximates the Max-Pressure policy
    from the traffic-signal control literature.

    Pass the current 8-lane queue array via `queues` to use pressure logic;
    falls back to keep-phase if queues is None.
    """
    def __init__(self, pressure_threshold: float = 5.0, min_green: int = 5) -> None:
        self.threshold = pressure_threshold
        self.min_green = min_green

    def act(self, phase_timer: int, queues=None) -> int:
        if queues is None or phase_timer < self.min_green:
            return 0
        p0 = float(np.sum(queues[:4]))
        p1 = float(np.sum(queues[4:]))
        return 1 if abs(p0 - p1) > self.threshold else 0