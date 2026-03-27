from __future__ import annotations

from typing import List, Dict
import numpy as np

from environment import TrafficEnvironment, EnvConfig
from dqn_agent import DQNAgent, DQNConfig
from utils import ensure_dir, save_training_plot, save_csv


def build_demand_profile(length: int) -> np.ndarray:
    x = np.linspace(0.8, 1.4, length // 2)
    y = np.linspace(1.4, 0.9, length - len(x))
    return np.concatenate([x, y]).astype(np.float32)


def train_one_seed(seed: int, episodes: int = 100, results_dir: str = "results") -> Dict[str, float]:
    ensure_dir(results_dir)

    env = TrafficEnvironment(
        cfg=EnvConfig(
            max_steps=200,
            min_green=5,
            reward_scale=0.01,
            use_external_factors=True,
            normalize_state=True,
        ),
        seed=seed,
        demand_profile=build_demand_profile(200),
    )

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        cfg=DQNConfig(
            gamma=0.99,
            learning_rate=1e-3,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.97,
            replay_capacity=50_000,
            batch_size=64,
            train_start=2_000,
            target_update_every_steps=1_000,
            use_double_dqn=True,
        ),
    )

    returns: List[float] = []
    episode_rows: List[Dict[str, float]] = []

    for ep in range(1, episodes + 1):
        state = env.reset().astype(np.float32).reshape(1, -1)

        ep_return = 0.0
        losses = []
        total_queue_last = 0.0
        cumulative_delay_last = 0.0

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32).reshape(1, -1)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_return += reward
            total_queue_last = info["total_queue"]
            cumulative_delay_last = info["cumulative_delay"]

        agent.end_episode()
        avg_loss = float(np.mean(losses)) if losses else float("nan")

        returns.append(ep_return)
        row = {
            "episode": float(ep),
            "return": float(ep_return),
            "avg_loss": avg_loss,
            "epsilon": float(agent.epsilon),
            "final_total_queue": float(total_queue_last),
            "cumulative_delay": float(cumulative_delay_last),
        }
        episode_rows.append(row)

        print(
            f"Seed {seed} | Episode {ep:03d} | "
            f"return={ep_return:10.2f} | "
            f"epsilon={agent.epsilon:0.3f} | "
            f"avg_loss={avg_loss:0.4f}"
        )

    model_path = "models/dqn_model_seed_{seed}.keras"
    plot_path = f"{results_dir}/training_curve_seed_{seed}.png"
    csv_path = f"{results_dir}/training_metrics_seed_{seed}.csv"

    agent.save(model_path)
    save_training_plot(returns, plot_path)
    save_csv(episode_rows, csv_path)

    last_10_mean = float(np.mean(returns[-10:]))
    return {
        "seed": float(seed),
        "last_10_return_mean": last_10_mean,
        "best_return": float(np.max(returns)),
        "final_return": float(returns[-1]),
    }


def main() -> None:
    seeds = [42, 123, 999]
    summary_rows = []

    for seed in seeds:
        summary = train_one_seed(seed=seed, episodes=100, results_dir="results")
        summary_rows.append(summary)

    save_csv(summary_rows, "results/seed_summary.csv")
    print("Saved results to ./results")


if __name__ == "__main__":
    main()
