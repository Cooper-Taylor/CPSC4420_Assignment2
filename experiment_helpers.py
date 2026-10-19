import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_learning import (
    TRAINING_SEEDS, VALIDATION_EPISODES, VALIDATION_SEED,
    train_q_learning, evaluate_policy, evaluation_metrics,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SMOOTHING_WINDOW = 100


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_setting(output_dir, label, settings):
    """Train one setting with the same seeds used by every experiment."""
    folder = output_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "config.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    reward_runs = []
    seed_summaries = []

    for seed in TRAINING_SEEDS:
        q_table, training = train_q_learning(seed=seed, **settings)
        validation = evaluate_policy(
            q_table, episodes=VALIDATION_EPISODES,
            seed=VALIDATION_SEED, action_noise=settings["action_noise"],
        )
        metrics = evaluation_metrics(validation)
        rewards = [row["reward"] for row in training]
        reward_runs.append(rewards)
        np.save(folder / f"q_table_seed_{seed}.npy", q_table)
        write_csv(folder / f"training_seed_{seed}.csv", training)
        write_csv(folder / f"validation_seed_{seed}.csv", validation)

        seed_summaries.append({
            "seed": seed,
            "training_mean_reward": float(np.mean(rewards)),
            "first_1000_mean_reward": float(np.mean(rewards[:1000])),
            "last_1000_mean_reward": float(np.mean(rewards[-1000:])),
            "last_1000_reward_std": float(np.std(rewards[-1000:])),
            **metrics,
        })
        print(f"{label}, seed {seed}: validation reward "
              f"{metrics['mean_reward']:.2f}, "
              f"success {metrics['success_rate']:.1%}", flush=True)

    write_csv(folder / "seed_summary.csv", seed_summaries)
    summary = {
        "label": label,
        **settings,
        "training_seeds": len(TRAINING_SEEDS),
        "validation_episodes_per_seed": VALIDATION_EPISODES,
        "validation_mean_reward": float(np.mean([
            row["mean_reward"] for row in seed_summaries
        ])),
        "validation_seed_reward_std": float(np.std([
            row["mean_reward"] for row in seed_summaries
        ])),
        "validation_success_rate": float(np.mean([
            row["success_rate"] for row in seed_summaries
        ])),
        "validation_mean_steps": float(np.mean([
            row["mean_steps"] for row in seed_summaries
        ])),
    }
    for metric in ["training_mean_reward", "first_1000_mean_reward",
                   "last_1000_mean_reward", "last_1000_reward_std"]:
        summary[metric] = float(np.mean([row[metric] for row in seed_summaries]))

    return {"summary": summary, "rewards": np.array(reward_runs)}


def save_comparison(output_dir, results, title):
    """Save learning curves and a table comparing the tested settings."""
    write_csv(output_dir / "summary.csv", [result["summary"] for result in results])
    fig, ax = plt.subplots(figsize=(9, 5))

    for result in results:
        rewards = result["rewards"]
        window = min(SMOOTHING_WINDOW, rewards.shape[1])
        smooth_runs = []
        for run in rewards:
            smooth_runs.append(np.convolve(run, np.ones(window) / window, mode="valid"))
        mean = np.mean(smooth_runs, axis=0)
        std = np.std(smooth_runs, axis=0)
        episodes = np.arange(window, rewards.shape[1] + 1)
        if len(results) == 1:
            ax.plot(np.arange(1, rewards.shape[1] + 1), rewards.mean(axis=0),
                    color="gray", alpha=0.22, linewidth=0.5, label="Raw seed mean")
        line, = ax.plot(episodes, mean, label=result["summary"]["label"])
        ax.fill_between(episodes, mean - std, mean + std,
                        color=line.get_color(), alpha=0.15)

    ax.set(title=title, xlabel="Training episode", ylabel="Episode reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01, "100-episode rolling means; shading: one standard deviation across training seeds",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "training_rewards.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [result["summary"]["label"] for result in results]
    means = [result["summary"]["validation_mean_reward"] for result in results]
    stds = [result["summary"]["validation_seed_reward_std"] for result in results]
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set(title=title + ": greedy validation", ylabel="Mean episode reward")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "100 episodes per trained policy; error bars: one standard deviation across training seeds",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "validation_rewards.png", dpi=180)
    plt.close(fig)
