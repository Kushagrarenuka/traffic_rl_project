from __future__ import annotations


class FixedTimeController:
    """
    Switch phase every fixed cycle length, respecting min-green.
    """

    def __init__(self, cycle_length: int = 10) -> None:
        self.cycle_length = cycle_length

    def act(self, phase_timer: int) -> int:
        if phase_timer >= self.cycle_length:
            return 1
        return 0