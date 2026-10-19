import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q_learning import BASELINE, train_q_learning, evaluate_policy, evaluation_metrics
from experiment_helpers import RESULTS, write_csv


EXPERIMENTS = ["baseline", "learning_rate", "discount_factor", "exploration_decay"]
FINAL_TRAINING_SEED = 42
TEST_EPISODES = 300
TEST_SEED = 20000


if __name__ == "__main__":
    candidates = []
    for experiment in EXPERIMENTS:
        path = RESULTS / experiment / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run baseline and all three parameter experiments first."
            )
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                row["experiment"] = experiment
                candidates.append(row)

    # Prefer validation reward; break ties using the overall training reward.
    best = candidates[0]
    for candidate in candidates[1:]:
        score = (float(candidate["validation_mean_reward"]),
                 float(candidate["training_mean_reward"]))
        best_score = (float(best["validation_mean_reward"]),
                      float(best["training_mean_reward"]))
        if score > best_score:
            best = candidate

    config_path = RESULTS / best["experiment"] / best["label"] / "config.json"
    with open(config_path, encoding="utf-8") as file:
        best_settings = json.load(file)

    output_dir = RESULTS / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "selected_experiment": best["experiment"],
        "selected_setting": best["label"],
        "selection_rule": "Highest mean validation reward across three training seeds; "
                          "ties use highest mean reward over training.",
        "validation_mean_reward": float(best["validation_mean_reward"]),
        "training_mean_reward": float(best["training_mean_reward"]),
        "baseline_settings": BASELINE,
        "best_settings": best_settings,
        "final_training_seed": FINAL_TRAINING_SEED,
        "test_episodes": TEST_EPISODES,
        "test_seed": TEST_SEED,
    }
    with open(output_dir / "selection.json", "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    print(f"Selected {best['experiment']} / {best['label']}", flush=True)
    summaries = []
    for label, settings in [("baseline", BASELINE), ("best", best_settings)]:
        # Retrain both with a fresh, matched seed before the final test.
        q_table, training = train_q_learning(seed=FINAL_TRAINING_SEED, **settings)
        evaluation = evaluate_policy(q_table, episodes=TEST_EPISODES, seed=TEST_SEED)
        np.save(output_dir / f"{label}_q_table.npy", q_table)
        write_csv(output_dir / f"{label}_training.csv", training)
        write_csv(output_dir / f"{label}_evaluation.csv", evaluation)
        metrics = evaluation_metrics(evaluation)
        summaries.append({"agent": label, "episodes": TEST_EPISODES, **metrics})
        print(f"{label}: reward {metrics['mean_reward']:.2f}, "
              f"success {metrics['success_rate']:.1%}, "
              f"steps {metrics['mean_steps']:.2f}", flush=True)

    write_csv(output_dir / "summary.csv", summaries)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric, title in zip(
        axes, ["mean_reward", "success_rate", "mean_steps"],
        ["Mean episode reward", "Success rate", "Mean steps"],
    ):
        ax.bar([row["agent"] for row in summaries], [row[metric] for row in summaries])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric == "success_rate":
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"Taxi-v3: {TEST_EPISODES} new greedy test episodes")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)
