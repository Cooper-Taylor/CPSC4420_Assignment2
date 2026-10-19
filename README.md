## Setup
Use Python 3.10 or newer. From this project's folder, run:

```bash
python -m venv .venv
source .venv/bin/activate
```

Pip install the required libraries too.

## Run the scripts

Run these commands in this order:

```bash
python baseline.py
python learning_rate_experiment.py
python discount_factor_experiment.py
python exploration_decay_experiment.py
python evaluate_best.py
python record_video.py
```

The three parameter experiment scripts can run in any order after the baseline.
Each script writes to its own folder under `results/`. Running a script again
replaces its outputs. After changing settings, rerun the experiments, then
`evaluate_best.py`, the video script.
