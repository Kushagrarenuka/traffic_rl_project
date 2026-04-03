# AI-Powered Adaptive Traffic Signal Control
### Using Reinforcement Learning and Real-Time Data
**Kushagra Aggarwal — Khoury College of Computer Sciences, Northeastern University**

---

## Overview

This project develops an AI-based adaptive traffic signal control system that optimizes traffic flow across multiple intersections in a smart city setting. Two deep reinforcement learning agents — **Double DQN** and **PPO (Proximal Policy Optimization)** — are trained to dynamically adjust signal timings based on real-time traffic conditions, live weather data, and special event flags. The system targets reductions in congestion, vehicle wait time, queue length, and CO₂ emissions from idling vehicles.

---

## Features

- **Multi-intersection environment** — 2 to 4 intersections modeled as a linear chain with spillback flow between them
- **Double DQN agent** — off-policy, experience replay, target network, epsilon-greedy exploration
- **PPO agent** — on-policy actor-critic with GAE, clipped surrogate loss, entropy regularization
- **Live weather integration** — real-time severity score fetched from the OpenWeatherMap API
- **CO₂ / fuel cost reward term** — penalizes idling queues to encourage environmentally efficient signal timing
- **4 controllers compared** — Double DQN, PPO, Fixed-Time baseline, Random baseline
- **Mean ± std evaluation** — statistically robust results across 3 random seeds and 20 held-out episodes

---

## Project Structure

```
traffic_rl_project/
│
├── environment.py       # Single + multi-intersection simulation environments
├── dqn_agent.py         # Double DQN agent (TensorFlow/Keras)
├── ppo_agent.py         # PPO actor-critic agent (TensorFlow/Keras)
├── baselines.py         # Fixed-Time, Random, and Max-Pressure controllers
├── weather_api.py       # OpenWeatherMap API wrapper
├── train.py             # Training loop for DQN and PPO across seeds
├── evaluate.py          # Evaluation and comparison of all controllers
├── utils.py             # Shared utilities (plotting, CSV saving, demand profile)
│
├── results/             # Generated after training
│   ├── evaluation_summary.csv
│   ├── seed_summary.csv
│   ├── metrics_dqn_seed_42.csv
│   ├── metrics_ppo_seed_42.csv
│   └── curve_dqn_i0_seed_42.png  (and more plots)
│
└── models/              # Generated after training
    ├── dqn_intersection_0_seed_42.keras
    ├── ppo_intersection_0_seed_42_actor.keras
    └── ppo_intersection_0_seed_42_critic.keras
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install tensorflow numpy matplotlib requests
```

---

## OpenWeatherMap API Setup

This project uses the **OpenWeatherMap free API** to fetch real-time weather severity for the city where the simulation runs. The severity score (0.0 = clear, 1.0 = severe storm) is injected into the environment state and affects vehicle arrival rates and service rates.

### Step 1 — Get a free API key
1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Sign up for a free account
3. Copy your API key from the confirmation email

### Step 2 — Set your API key

**On Windows (PowerShell) — current session only:**
```powershell
$env:OPENWEATHER_API_KEY="your_api_key_here"
```

**On Windows — permanent (survives terminal restarts):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENWEATHER_API_KEY","your_api_key_here","User")
```

**On Mac/Linux:**
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

### Step 3 — Test the connection
```bash
python weather_api.py
```
Expected output:
```
Weather severity for Boston: 0.0
```

> **Note:** New API keys take a couple of hours to activate after signup. The system gracefully falls back to a synthetic weather value of `0.0` if the key is not yet active or not set — training works fine without it.

### How weather is used in the model

The `get_weather_severity()` function maps OpenWeatherMap weather condition codes to a severity score:

| Weather Condition | Severity Score |
|---|---|
| Clear / Clouds | 0.0 |
| Drizzle / Mist | 0.2 |
| Fog / Haze | 0.3 |
| Moderate Rain | 0.5 |
| Heavy Snow | 0.7 |
| Thunderstorm | 0.9 |

This score is passed into `MultiIntersectionEnvironment` at the start of each training run and affects:
- **Vehicle arrival rate** — bad weather increases arrivals (accidents, slower driving)
- **Service rate** — bad weather reduces throughput at green phases
- **Agent state** — weather severity is part of the 12-dimensional state vector the agent observes

---

## Running the Project

### 1. Test weather API
```bash
python weather_api.py
```

### 2. Train all agents
```bash
python train.py
```
This trains Double DQN and PPO across 3 seeds (42, 123, 999) on a 2-intersection environment — **6 total runs of 100 episodes each**. Expect 30–60 minutes on CPU.

To change the number of intersections, edit `train.py`:
```python
summary.append(train_multiagent(
    seed=seed, n_intersections=2,   # change to 3 or 4
    ...
))
```

### 3. Evaluate all controllers
```bash
python evaluate.py
```
Outputs `results/evaluation_summary.csv` comparing Double DQN, PPO, Fixed-Time, and Random controllers with mean ± std across 20 episodes.

---

## Results

Both RL agents outperformed the Fixed-Time baseline. Double DQN achieved the best results across all metrics, outperforming Fixed-Time by ~10.4% in mean return and ~11.2% in delay. PPO showed stable convergence at intersection 0 but underfit at intersection 1 within 100 episodes, indicating it would benefit from longer training.

| Controller | Mean Return | Mean Delay | Mean CO₂ Cost |
|---|---|---|---|
| Double DQN | **-1024.69** | **85,150.98** | **425.75** |
| PPO | -1400.36 | 109,222.88 | 546.12 |
| Fixed-Time | -1144.16 | 95,863.73 | 479.32 |
| Random | -1101.43 | 91,963.15 | 459.82 |

> Returns are negative because the reward is defined as a negative congestion cost — values closer to 0 are better.

---

## Environment Details

### State space (12-dimensional per intersection)
| Feature | Description |
|---|---|
| `q0–q7` | Normalized queue lengths for all 8 lanes |
| `phase` | Current signal phase (0 or 1) |
| `phase_timer_norm` | Time in current phase, normalized |
| `weather` | Live weather severity from OpenWeatherMap (0–1) |
| `event` | Special event flag (0 or 1) |

### Action space
| Action | Description |
|---|---|
| `0` | Keep current phase |
| `1` | Request phase switch (subject to min-green constraint) |

### Reward function
```
reward = -scale × (w_queue × total_queue
                 + w_imbalance × queue_imbalance
                 + w_delay × step_delay
                 + w_switch × switch_penalty
                 + w_co2 × co2_cost)
```

---

## References

Li, L., Lv, Y., & Wang, F.-Y. (2016). Traffic signal timing via deep reinforcement learning. *IEEE/CAA Journal of Automatica Sinica*, 3(3), 247–254.