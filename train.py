from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from environment import TrafficEnvironment, EnvConfig
from dqn_agent import DQNAgent, DQNConfig


def main() -> None:
    env = TrafficEnvironment(cfg=EnvConfig(max_steps=200, use_external_factors=True), seed=42)
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
        ),
    )

    episodes = 100
    max_steps = env.cfg.max_steps

    returns = []

    for ep in range(1, episodes + 1):
        state = env.reset().astype(np.float32)
        state = state.reshape(1, -1)

        ep_return = 0.0
        losses = []

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _info = env.step(action)

            next_state = next_state.astype(np.float32).reshape(1, -1)

            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_return += reward

            if done:
                break

        returns.append(ep_return)
        
        agent.end_episode()
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"Episode {ep:04d} | return={ep_return:10.2f} | epsilon={agent.epsilon:0.3f} | avg_loss={avg_loss:0.4f}"
        )

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Performance")
    plt.show()

    agent.save("./saved_dqn_model")
    print("Saved model to ./saved_dqn_model")


if __name__ == "__main__":
    main()