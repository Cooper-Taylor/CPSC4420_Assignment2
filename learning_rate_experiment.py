from q_learning import BASELINE
from experiment_helpers import RESULTS, run_setting, save_comparison


ALPHA_VALUES = [0.1, 0.5, 0.9]


if __name__ == "__main__":
    output_dir = RESULTS / "learning_rate"
    results = []
    for alpha in ALPHA_VALUES:
        settings = BASELINE.copy()
        settings["alpha"] = alpha
        results.append(run_setting(output_dir, f"alpha_{alpha}", settings))
    save_comparison(output_dir, results, "Taxi-v3: learning rate")
