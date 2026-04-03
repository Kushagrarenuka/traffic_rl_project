from __future__ import annotations

from typing import Dict, List, Literal
import numpy as np

from environment  import MultiIntersectionEnvironment, MultiEnvConfig
from dqn_agent    import DQNAgent, DQNConfig
from ppo_agent    import PPOAgent
from baselines    import FixedTimeController, RandomController
from weather_api  import get_weather_severity
from utils        import ensure_dir, build_demand_profile, save_csv

AgentType = Literal["dqn", "ppo"]


def _load_agent(kind: AgentType, state_size: int, action_size: int, path: str):
    agent = PPOAgent(state_size, action_size) if kind == "ppo" else DQNAgent(state_size, action_size, DQNConfig())
    agent.load(path)
    return agent


def _make_env(n: int, seed: int, city: str) -> MultiIntersectionEnvironment:
    return MultiIntersectionEnvironment(
        cfg=MultiEnvConfig(n_intersections=n, max_steps=200, min_green=5,
                           use_external_factors=True, normalize_state=True, w_co2=0.10),
        seed=seed,
        demand_profile=build_demand_profile(200),
        weather_severity=get_weather_severity(city=city),
    )


def _run_episodes(env, n, act_fn, episodes) -> Dict:
    """Run `episodes` greedy episodes; return mean ± std for key metrics."""
    rets   = [[] for _ in range(n)]
    delays = [[] for _ in range(n)]
    co2s   = [[] for _ in range(n)]

    for _ in range(episodes):
        raw    = env.reset()
        states = [s.astype(np.float32).reshape(1, -1) for s in raw]
        ep_ret = [0.0] * n
        done   = False
        while not done:
            actions          = act_fn(states, env)
            next_raw, rews, done, infos = env.step(actions)
            states = [s.astype(np.float32).reshape(1, -1) for s in next_raw]
            for i in range(n):
                ep_ret[i] += rews[i]
        for i in range(n):
            rets[i].append(ep_ret[i])
            delays[i].append(infos[i]["cumulative_delay"])
            co2s[i].append(infos[i]["total_co2"])

    row = {}
    for i in range(n):
        row[f"mean_return_i{i}"] = float(np.mean(rets[i]))
        row[f"std_return_i{i}"]  = float(np.std(rets[i]))
        row[f"mean_delay_i{i}"]  = float(np.mean(delays[i]))
        row[f"mean_co2_i{i}"]    = float(np.mean(co2s[i]))
    return row


def evaluate_rl(kind: AgentType, model_paths: List[str], n: int = 2,
                episodes: int = 20, seed: int = 777, city: str = "Boston") -> Dict:
    env    = _make_env(n, seed, city)
    agents = [_load_agent(kind, env.state_size, env.action_size, p) for p in model_paths]

    def act_fn(states, _env):
        actions = []
        for i, ag in enumerate(agents):
            a = ag.act(states[i], greedy=True)
            actions.append(a[0] if kind == "ppo" else a)
        return actions

    return {"controller": kind, **_run_episodes(env, n, act_fn, episodes)}


def evaluate_baselines(n: int = 2, episodes: int = 20,
                       seed: int = 777, city: str = "Boston") -> List[Dict]:
    rows = []
    for name, make_ctrl in [
        ("fixed_time", lambda: [FixedTimeController(cycle_length=10) for _ in range(n)]),
        ("random",     lambda: [RandomController() for _ in range(n)]),
    ]:
        env   = _make_env(n, seed, city)
        ctrls = make_ctrl()

        def act_fn(states, _env, _ctrls=ctrls):
            return [_ctrls[i].act(_env.phase_timers[i]) for i in range(n)]

        rows.append({"controller": name, **_run_episodes(env, n, act_fn, episodes)})
    return rows


def main() -> None:
    ensure_dir("results")
    n    = 2
    rows = []
    rows.append(evaluate_rl("dqn", [f"models/dqn_intersection_{i}_seed_42.keras" for i in range(n)], n=n))
    rows.append(evaluate_rl("ppo", [f"models/ppo_intersection_{i}_seed_42" for i in range(n)], n=n))
    rows.extend(evaluate_baselines(n=n))
    save_csv(rows, "results/evaluation_summary.csv")
    print("Evaluation saved to ./results/evaluation_summary.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()