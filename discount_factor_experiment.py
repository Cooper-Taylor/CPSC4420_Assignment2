from q_learning import BASELINE
from experiment_helpers import RESULTS, run_setting, save_comparison


GAMMA_VALUES = [0.50, 0.90, 0.99]


if __name__ == "__main__":
    output_dir = RESULTS / "discount_factor"
    results = []
    for gamma in GAMMA_VALUES:
        settings = BASELINE.copy()
        settings["gamma"] = gamma
        results.append(run_setting(output_dir, f"gamma_{gamma}", settings))
    save_comparison(output_dir, results, "Taxi-v3: discount factor")
