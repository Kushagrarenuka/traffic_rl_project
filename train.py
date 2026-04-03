from __future__ import annotations

import os
from typing import Dict, List, Literal
import numpy as np

from environment import MultiIntersectionEnvironment, MultiEnvConfig
from dqn_agent  import DQNAgent, DQNConfig
from ppo_agent  import PPOAgent, PPOConfig
from weather_api import get_weather_severity
from utils import ensure_dir, build_demand_profile, save_training_plot, save_csv

AgentType = Literal["dqn", "ppo"]


def _make_agents(n: int, state_size: int, action_size: int, kind: AgentType):
    """Instantiate one fresh agent per intersection."""
    agents = []
    for _ in range(n):
        if kind == "ppo":
            agents.append(PPOAgent(state_size=state_size, action_size=action_size))
        else:
            agents.append(DQNAgent(
                state_size=state_size, action_size=action_size,
                cfg=DQNConfig(
                    gamma=0.99, learning_rate=1e-3,
                    epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97,
                    replay_capacity=50_000, batch_size=64, train_start=2_000,
                    target_update_every_steps=1_000, use_double_dqn=True,
                ),
            ))
    return agents


def train_multiagent(
    seed: int,
    n_intersections: int  = 2,
    agent_type: AgentType = "dqn",
    episodes: int         = 100,
    results_dir: str      = "results",
    city: str             = "Boston",
) -> Dict:
    """
    Train one independent agent per intersection.

    PROPOSAL alignments addressed here
    -----------------------------------
    - Multi-agent (n_intersections ≥ 2)
    - Real OpenWeather API weather severity
    - CO2 term active in MultiEnvConfig (w_co2=0.10)
    - PPO as alternative to Double DQN
    - BUG FIX: f-string was missing 'f' prefix in original train.py
    """
    ensure_dir(results_dir)
    ensure_dir("models")   # BUG FIX: original code would crash if models/ absent

    # ── live weather ──────────────────────────────────────────────────────────
    weather_sev = get_weather_severity(city=city)
    print(f"[WeatherAPI] {city} severity = {weather_sev:.3f}")

    env = MultiIntersectionEnvironment(
        cfg=MultiEnvConfig(
            n_intersections=n_intersections,
            max_steps=200, min_green=5, reward_scale=0.01,
            use_external_factors=True, normalize_state=True, w_co2=0.10,
        ),
        seed=seed,
        demand_profile=build_demand_profile(200),
        weather_severity=weather_sev,
    )

    agents = _make_agents(n_intersections, env.state_size, env.action_size, agent_type)

    all_returns: List[List[float]] = [[] for _ in range(n_intersections)]
    episode_rows: List[Dict]       = []

    for ep in range(1, episodes + 1):
        raw    = env.reset()
        states = [s.astype(np.float32).reshape(1, -1) for s in raw]
        ep_ret = [0.0] * n_intersections
        ep_co2 = [0.0] * n_intersections
        done   = False

        while not done:
            actions, log_probs, values = [], [], []
            for i, agent in enumerate(agents):
                if agent_type == "ppo":
                    a, lp, v = agent.act(states[i])
                    log_probs.append(lp); values.append(v)
                else:
                    a = agent.act(states[i])
                actions.append(a)

            next_raw, rewards, done, infos = env.step(actions)
            nxt = [s.astype(np.float32).reshape(1, -1) for s in next_raw]

            for i, agent in enumerate(agents):
                if agent_type == "ppo":
                    agent.store(states[i], actions[i], rewards[i], values[i], log_probs[i], done)
                else:
                    agent.remember(states[i], actions[i], rewards[i], nxt[i], done)
                    agent.train_step()
                ep_ret[i] += rewards[i]
                ep_co2[i] += infos[i]["co2_step"]

            states = nxt

        # ── end-of-episode ────────────────────────────────────────────────────
        if agent_type == "ppo":
            for agent in agents:
                agent.train(last_value=0.0)
        else:
            for agent in agents:
                agent.end_episode()

        for i in range(n_intersections):
            all_returns[i].append(ep_ret[i])

        mean_ret = float(np.mean(ep_ret))
        row = {"episode": ep, "agent_type": agent_type, "mean_return": mean_ret,
               "weather_severity": weather_sev,
               **{f"return_i{i}": ep_ret[i] for i in range(n_intersections)},
               **{f"co2_i{i}":    ep_co2[i]  for i in range(n_intersections)}}
        episode_rows.append(row)
        print(f"Seed {seed} | Ep {ep:03d} | mean_ret={mean_ret:9.2f} | "
              f"weather={weather_sev:.2f} | {agent_type.upper()}")

    # ── save ──────────────────────────────────────────────────────────────────
    for i, agent in enumerate(agents):
        # BUG FIX: added f-prefix (original was "models/dqn_model_seed_{seed}.keras" literally)
        path = f"models/{agent_type}_intersection_{i}_seed_{seed}.keras" if agent_type == "dqn" else f"models/{agent_type}_intersection_{i}_seed_{seed}"
        agent.save(path)

    for i in range(n_intersections):
        save_training_plot(all_returns[i],
                           f"{results_dir}/curve_{agent_type}_i{i}_seed_{seed}.png")
    save_csv(episode_rows, f"{results_dir}/metrics_{agent_type}_seed_{seed}.csv")

    return {"seed": seed, "agent_type": agent_type, "n_intersections": n_intersections,
            **{f"last10_i{i}": float(np.mean(all_returns[i][-10:])) for i in range(n_intersections)}}


def main() -> None:
    seeds   = [42, 123, 999]
    summary = []
    for kind in ("dqn", "ppo"):           # compare both agents  (proposal requirement)
        for seed in seeds:
            summary.append(train_multiagent(
                seed=seed, n_intersections=2,   # change to 3 or 4 to scale up
                agent_type=kind, episodes=100,
                results_dir="results", city="Boston",
            ))
    save_csv(summary, "results/seed_summary.csv")
    print("Done — results in ./results/")


if __name__ == "__main__":
    main()