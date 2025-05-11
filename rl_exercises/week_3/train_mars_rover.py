from __future__ import annotations

import pandas as pd
from rl_exercises.environments import MarsRover
from rl_exercises.week_3 import EpsilonGreedyPolicy, SARSAAgent


def train_agent(env, agent, save_path, training_steps):
    for _ in range(training_steps):
        state = env.position
        action = agent.predict_action(state)  # take an action based on current state
        next_state, reward, terminated, truncated, info = env.step(action)
        next_action = agent.predict_action(next_state)

        while True:
            agent.update_agent(
                env.position, action, reward, next_state, next_action, truncated
            )
            state = next_state
            action = next_action

            if truncated:
                break

            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = agent.predict_action(next_state)

        model = pd.DataFrame(agent.Q)
        model.to_csv(save_path, mode="a")

        env.reset()


if __name__ == "__main__":
    training_steps = 10

    # optimal parameters
    opt_env = MarsRover(rewards=[5, 1, 0, 0, 10], horizon=2)
    opt_policy = EpsilonGreedyPolicy(opt_env, epsilon=0.88)
    opt_agent = SARSAAgent(opt_env, opt_policy, alpha=0.72, gamma=0.57)
    opt_model_path = (
        "/home/sarah/repos/rl/rl-week-3-rl_group3/rl_exercises/week_3/opt_model.csv"
    )

    train_agent(opt_env, opt_agent, opt_model_path, training_steps)

    # random parameters
    rnd_env = MarsRover(rewards=[5, 1, 0, 0, 10], horizon=5)
    rnd_policy = EpsilonGreedyPolicy(rnd_env, epsilon=0.5)
    rnd_agent = SARSAAgent(rnd_env, rnd_policy, alpha=0.8, gamma=0.4)
    rnd_model_path = (
        "/home/sarah/repos/rl/rl-week-3-rl_group3/rl_exercises/week_3/rnd_model.csv"
    )

    train_agent(rnd_env, rnd_agent, rnd_model_path, training_steps)
