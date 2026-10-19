from q_learning import BASELINE
from experiment_helpers import RESULTS, run_setting, save_comparison


if __name__ == "__main__":
    output_dir = RESULTS / "baseline"
    result = run_setting(output_dir, "baseline", BASELINE.copy())
    save_comparison(output_dir, [result], "Taxi-v3: baseline")
