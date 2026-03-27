from __future__ import annotations

from typing import Dict, List
import numpy as np
import tensorflow as tf

from baselines import FixedTimeController
from environment import TrafficEnvironment, EnvConfig
from dqn_agent import DQNAgent, DQNConfig
from utils import save_csv, ensure_dir


def build_demand_profile(length: int) -> np.ndarray:
    x = np.linspace(0.8, 1.4, length // 2)
    y = np.linspace(1.4, 0.9, length - len(x))
    return np.concatenate([x, y]).astype(np.float32)


def evaluate_dqn(model_path: str, episodes: int = 20, seed: int = 777) -> Dict[str, float]:
    env = TrafficEnvironment(
        cfg=EnvConfig(max_steps=200, min_green=5, use_external_factors=True, normalize_state=True),
        seed=seed,
        demand_profile=build_demand_profile(200),
    )
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        cfg=DQNConfig(),
    )
    agent.load(model_path)

    returns = []
    delays = []
    queues = []

    for _ in range(episodes):
        state = env.reset().astype(np.float32).reshape(1, -1)
        done = False
        ep_return = 0.0
        final_delay = 0.0
        final_queue = 0.0

        while not done:
            action = agent.act(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            state = next_state.astype(np.float32).reshape(1, -1)
            ep_return += reward
            final_delay = info["cumulative_delay"]
            final_queue = info["total_queue"]

        returns.append(ep_return)
        delays.append(final_delay)
        queues.append(final_queue)

    return {
        "controller": "double_dqn",
        "mean_return": float(np.mean(returns)),
        "mean_delay": float(np.mean(delays)),
        "mean_final_queue": float(np.mean(queues)),
    }


def evaluate_fixed_time(episodes: int = 20, seed: int = 777) -> Dict[str, float]:
    env = TrafficEnvironment(
        cfg=EnvConfig(max_steps=200, min_green=5, use_external_factors=True, normalize_state=True),
        seed=seed,
        demand_profile=build_demand_profile(200),
    )
    controller = FixedTimeController(cycle_length=10)

    returns = []
    delays = []
    queues = []

    for _ in range(episodes):
        env.reset()
        done = False
        ep_return = 0.0
        final_delay = 0.0
        final_queue = 0.0

        while not done:
            action = controller.act(env.phase_timer)
            _, reward, done, info = env.step(action)
            ep_return += reward
            final_delay = info["cumulative_delay"]
            final_queue = info["total_queue"]

        returns.append(ep_return)
        delays.append(final_delay)
        queues.append(final_queue)

    return {
        "controller": "fixed_time",
        "mean_return": float(np.mean(returns)),
        "mean_delay": float(np.mean(delays)),
        "mean_final_queue": float(np.mean(queues)),
    }


def main() -> None:
    ensure_dir("results")
    rows: List[Dict[str, float]] = []

    dqn_result = evaluate_dqn("models/dqn_model_seed_42.keras", episodes=20, seed=777)
    baseline_result = evaluate_fixed_time(episodes=20, seed=777)

    rows.append(dqn_result)
    rows.append(baseline_result)

    save_csv(rows, "results/evaluation_summary.csv")
    print("Saved evaluation to ./results/evaluation_summary.csv")
    print(rows)


if __name__ == "__main__":
    main()