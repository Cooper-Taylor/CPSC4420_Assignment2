import gymnasium as gym
import numpy as np


ENV_NAME = "Taxi-v3"
TRAINING_SEEDS = [0, 1, 2]
VALIDATION_EPISODES = 100
VALIDATION_SEED = 10000

# The experiment scripts copy these settings and change one parameter for easy variability!
BASELINE = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.999,
    "episodes": 10000,
    "action_noise": 0.0,
}


def add_action_noise(action, number_of_actions, probability, rng):
    if probability > 0 and rng.random() < probability:
        other_actions = list(range(number_of_actions))
        other_actions.remove(action)
        return int(rng.choice(other_actions))
    return action


def train_q_learning(alpha=0.1, gamma=0.99, epsilon_start=1.0,
                     epsilon_min=0.01, epsilon_decay=0.999, episodes=10000,
                     seed=0, action_noise=0.0):
    env = gym.make(ENV_NAME)
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 100000)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = epsilon_start
    history = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total_reward = 0
        steps = 0
        illegal_actions = 0
        done = False

        while not done:
            # Epsilon-greedy action selection, using all six actions.
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q_table[state]))

            executed_action = add_action_noise(
                action, env.action_space.n, action_noise, noise_rng
            )
            next_state, reward, terminated, truncated, _ = env.step(executed_action)

            # A real terminal state has no future reward. A time limit does.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            # Update the intended action: action noise is part of the dynamics.
            q_table[state, action] += alpha * (target - q_table[state, action])

            state = next_state
            total_reward += reward
            steps += 1
            if reward == -10:
                illegal_actions += 1
            done = terminated or truncated

        history.append({
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "success": int(terminated),
            "truncated": int(truncated),
            "illegal_actions": illegal_actions,
            "epsilon": epsilon,
        })
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return q_table, history


def evaluate_policy(q_table, episodes=300, seed=20000, action_noise=0.0):
    env = gym.make(ENV_NAME)
    history = []

    for episode in range(episodes):
        # The same episode seeds give each policy the same starting tasks.
        state, _ = env.reset(seed=seed + episode)
        noise_rng = np.random.default_rng(seed + episode + 100000)
        total_reward = 0
        steps = 0
        illegal_actions = 0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            executed_action = add_action_noise(
                action, env.action_space.n, action_noise, noise_rng
            )
            state, reward, terminated, truncated, _ = env.step(executed_action)
            total_reward += reward
            steps += 1
            if reward == -10:
                illegal_actions += 1
            done = terminated or truncated

        history.append({
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "success": int(terminated),
            "truncated": int(truncated),
            "illegal_actions": illegal_actions,
        })

    env.close()
    return history


def evaluation_metrics(history):
    successful_steps = [row["steps"] for row in history if row["success"]]
    return {
        "mean_reward": float(np.mean([row["reward"] for row in history])),
        "reward_std": float(np.std([row["reward"] for row in history])),
        "success_rate": float(np.mean([row["success"] for row in history])),
        "mean_steps": float(np.mean([row["steps"] for row in history])),
        "mean_successful_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "mean_illegal_actions": float(np.mean([
            row["illegal_actions"] for row in history
        ])),
    }
