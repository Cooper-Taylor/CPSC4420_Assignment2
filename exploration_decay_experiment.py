from q_learning import BASELINE
from experiment_helpers import RESULTS, run_setting, save_comparison


DECAY_VALUES = [0.995, 0.999, 0.9995]


if __name__ == "__main__":
    output_dir = RESULTS / "exploration_decay"
    results = []
    for decay in DECAY_VALUES:
        settings = BASELINE.copy()
        settings["epsilon_decay"] = decay
        results.append(run_setting(output_dir, f"decay_{decay}", settings))
    save_comparison(output_dir, results, "Taxi-v3: exploration decay")
